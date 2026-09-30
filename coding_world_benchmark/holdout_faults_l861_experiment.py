"""L8.6.1: hold-out self-repair over faults the diagnoser was not built against.

L8.6 designs fixtures and diagnosis heuristics together, so its score measures
whether the loop closes on *known* fault shapes.  This module measures the gap
to unknown ones.

The generator is deliberately different.  Each hold-out starts from a module
that is correct -- its probe and baseline both pass -- and applies one explicit
textual corruption.  That gives every fixture a known-good oracle, guarantees
the fault is a real regression of working code rather than a hand-written
puzzle, and keeps the break auditable: the corruption is a visible
``(before, after)`` pair, not prose.

The fault shapes are the same capability categories as L8.6 (execute, move,
research, summarize, math, repair) broken differently: values computed and
dropped, a condition inverted, an argument omitted, a filter skipped, an
abstraction step missing, a verification step missing.  These are *intra-body
dataflow* faults, where L8.6's faults were *wiring* faults.

Protocol: the diagnoser is frozen before these fixtures are written, and is not
tuned against them afterwards.  Any later improvement aimed at these faults
turns them into known faults and needs a fresh hold-out to re-measure.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .external_repair_l81_experiment import RegressionVerifier
from .self_modification_l86_experiment import (
    BASELINE,
    FAULT_FIXTURES,
    MODULE,
    PROBE,
    BenchmarkResult,
    ExpectedAction,
    FaultClass,
    FaultFixture,
    FixtureOutcome,
    build_fixture,
    format_self_modification_report,
    observe_fault,
    run_fixture,
    run_self_modification_benchmark,
    summarize,
)


_FALLBACK = (
    "def keyword_search_fallback(request):\n"
    '    return "search-hit: " + request\n'
)

_DISPATCH = (
    "def handle_request(request):\n"
    "    _routed = {route}(request)\n"
    "    if _routed is not None:\n"
    "        return _routed\n"
    "    return keyword_search_fallback(request)\n"
)

_UNRELATED_BASELINE = (
    "from agent_module import handle_request\n\n\n"
    "def test_unrelated_request_keeps_previous_behaviour():\n"
    '    assert handle_request("unrelated request").startswith("search-hit:")\n'
)


@dataclass(frozen=True)
class HoldoutFixture:
    """A correct module plus one auditable corruption."""

    id: str
    capability: str
    break_shape: str
    healthy_module: str
    corruption: tuple[str, str]
    probe: str
    baseline: str

    @property
    def broken_module(self) -> str:
        before, after = self.corruption
        if before not in self.healthy_module:
            raise ValueError(f"{self.id}: corruption target not found in the healthy module")
        return self.healthy_module.replace(before, after, 1)

    def as_fault_fixture(self, healthy: bool = False) -> FaultFixture:
        """A hold-out is repairable in principle, so its expected action is REPAIR.

        ``expected_fault`` is UNKNOWN: it records that the L8.6 taxonomy has no
        entry for this shape, so ``diagnosis_accuracy`` over hold-outs reads as
        "how often did it correctly recognize that no detector applies".
        """
        return FaultFixture(
            self.id, self.capability, self.break_shape, FaultClass.UNKNOWN,
            {
                MODULE: self.healthy_module if healthy else self.broken_module,
                PROBE: self.probe,
                BASELINE: self.baseline,
            },
            ExpectedAction.IGNORE if healthy else ExpectedAction.REPAIR,
        )


def _probe(request: str, body: str) -> str:
    return (
        "from agent_module import handle_request\n\n\n"
        "def test_capability_produces_the_required_output():\n"
        f'    result = handle_request("{request}")\n'
        f"{body}"
    )


H1 = HoldoutFixture(
    "H1", "EXECUTE", "subprocess の結果を受け取るが捨てている",
    (
        '"""Execute capability: runs the command and reports its output."""\n\n\n'
        "def run_command(request):\n"
        '    return "output-of: " + request\n\n\n'
        + _FALLBACK +
        "\n\ndef execute_route(request):\n"
        '    if "実行" in request:\n'
        "        result = run_command(request)\n"
        '        return "executed: " + result\n'
        "    return None\n\n\n"
        + _DISPATCH.format(route="execute_route")
    ),
    ('        return "executed: " + result', '        return "executed"'),
    _probe("テストを実行して", '    assert result.startswith("executed:")\n    assert "output-of" in result\n'),
    _UNRELATED_BASELINE,
)

H2 = HoldoutFixture(
    "H2", "MOVE_DIRECTORIES", "移動は成功するが検証条件が反転している",
    (
        '"""Move capability: moves the directory and verifies the result."""\n\n\n'
        "def move_directory(source, destination):\n"
        '    return {"source_exists": False, "destination_exists": True}\n\n\n'
        "def verify_move(state):\n"
        '    return state["destination_exists"] and not state["source_exists"]\n\n\n'
        "def handle_request(request):\n"
        '    state = move_directory("SU2", "su2_su3")\n'
        '    return "moved" if verify_move(state) else "move-failed"\n'
    ),
    ('    return state["destination_exists"] and not state["source_exists"]',
     '    return state["source_exists"] and not state["destination_exists"]'),
    _probe("SU2のフォルダを移動して", '    assert result == "moved"\n'),
    (
        "from agent_module import move_directory\n\n\n"
        "def test_move_contract_is_unchanged():\n"
        '    assert move_directory("a", "b")["destination_exists"] is True\n'
    ),
)

H3 = HoldoutFixture(
    "H3", "SCIENTIFIC_RESEARCH", "evidence を取得するが planner へ渡していない",
    (
        '"""Research capability: retrieves evidence and plans with it."""\n\n\n'
        "def fetch_evidence(request):\n"
        '    return "evidence: " + request\n\n\n'
        "def build_plan(request, evidence=\"\"):\n"
        '    return "plan: " + request + ((" | " + evidence) if evidence else "")\n\n\n'
        + _FALLBACK +
        "\n\ndef research_route(request):\n"
        '    if "調査" in request:\n'
        "        evidence = fetch_evidence(request)\n"
        "        return build_plan(request, evidence)\n"
        "    return None\n\n\n"
        + _DISPATCH.format(route="research_route")
    ),
    ("        return build_plan(request, evidence)", "        return build_plan(request)"),
    _probe("先行研究を調査して", '    assert result.startswith("plan:")\n    assert "evidence:" in result\n'),
    _UNRELATED_BASELINE,
)

H4 = HoldoutFixture(
    "H4", "SUMMARIZE", "ContextGraph は使うが無関係 node を絞り込まない",
    (
        '"""Summarize capability: filters the context graph before summarizing."""\n\n\n'
        "def context_graph(request):\n"
        '    return ["core-node", "unrelated-1", "unrelated-2"]\n\n\n'
        "def relevant_nodes(nodes):\n"
        '    return [item for item in nodes if not item.startswith("unrelated")]\n\n\n'
        + _FALLBACK +
        "\n\ndef summarize_route(request):\n"
        '    if "要約" in request:\n'
        '        return "summary: " + ", ".join(relevant_nodes(context_graph(request)))\n'
        "    return None\n\n\n"
        + _DISPATCH.format(route="summarize_route")
    ),
    ('", ".join(relevant_nodes(context_graph(request)))', '", ".join(context_graph(request))'),
    _probe("これを要約して", '    assert result.startswith("summary:")\n    assert "unrelated" not in result\n'),
    _UNRELATED_BASELINE,
)

H5 = HoldoutFixture(
    "H5", "MATH_DERIVATION", "離散更新式は返すが governing equation へ抽象化しない",
    (
        '"""Math capability: reports the discrete update and the governing equation."""\n\n\n'
        "def discrete_update(request):\n"
        '    return "update: u_next = 2*u - u_prev"\n\n\n'
        "def governing_equation(request):\n"
        '    return "equation: d2u/dt2 = c**2 * d2u/dx2"\n\n\n'
        + _FALLBACK +
        "\n\ndef math_route(request):\n"
        '    if "導出" in request:\n'
        '        return discrete_update(request) + " | " + governing_equation(request)\n'
        "    return None\n\n\n"
        + _DISPATCH.format(route="math_route")
    ),
    ('        return discrete_update(request) + " | " + governing_equation(request)',
     "        return discrete_update(request)"),
    _probe("波動方程式を導出して", '    assert "update:" in result\n    assert "equation:" in result\n'),
    _UNRELATED_BASELINE,
)

H6 = HoldoutFixture(
    "H6", "EXTERNAL_REPAIR_PROPOSAL", "patch は当てるが regression を実行しない",
    (
        '"""Repair capability: applies the patch and runs the regression suite."""\n\n\n'
        "def apply_patch(request):\n"
        '    return "patch-applied"\n\n\n'
        "def run_regression():\n"
        '    return "regression: 1 passed"\n\n\n'
        + _FALLBACK +
        "\n\ndef repair_route(request):\n"
        '    if "修正" in request:\n'
        '        return "repaired: " + apply_patch(request) + " | " + run_regression()\n'
        "    return None\n\n\n"
        + _DISPATCH.format(route="repair_route")
    ),
    ('        return "repaired: " + apply_patch(request) + " | " + run_regression()',
     '        return "repaired: " + apply_patch(request)'),
    _probe("この提案で修正して", '    assert result.startswith("repaired:")\n    assert "regression:" in result\n'),
    _UNRELATED_BASELINE,
)

HOLDOUT_FIXTURES: tuple[HoldoutFixture, ...] = (H1, H2, H3, H4, H5, H6)

# The uncorrupted H4 is the hold-out set's healthy control.  It is the strongest
# one available here: in the corrupted version the generic unwired rule misfires
# on `relevant_nodes`, and in this version that same helper is referenced, so a
# detector that keys on "unused helper" must stay silent.
HOLDOUT_CONTROL = H4


# Recorded on the first (cold) run, before any tuning against these fixtures.
# container / Python 3.11 / 2026-09-14.  A change here means the hold-out has
# been burned: mint a fresh one rather than editing this record.
COLD_MEASUREMENT = {
    "repair_success_rate": 0.0,
    "fault_detection_rate": 1.0,
    "false_repair_count": 0,
    "breakdown": {
        "H1": "no_applicable_detector",
        "H2": "no_applicable_detector",
        "H3": "no_applicable_detector",
        "H4": "misdiagnosed_then_rejected_by_regression_gate",
        "H5": "misdiagnosed_then_rejected_by_regression_gate",
        "H6": "misdiagnosed_then_rejected_by_regression_gate",
        "H0": "undetected",
    },
}


def verify_holdout_is_well_formed(root: Path, fixture: HoldoutFixture) -> tuple[bool, bool, bool]:
    """(healthy_passes, broken_probe_fails, broken_baseline_passes).

    A hold-out is only usable when the uncorrupted module passes everything, the
    corruption breaks the probe, and it leaves the baseline intact -- otherwise
    the fixture tests more than one thing.
    """
    healthy_root = root / "healthy"
    broken_root = root / "broken"
    build_fixture(healthy_root, fixture.as_fault_fixture(healthy=True))
    build_fixture(broken_root, fixture.as_fault_fixture())
    healthy = not observe_fault(healthy_root, (PROBE, BASELINE)).detected
    probe_fails = observe_fault(broken_root, (PROBE,)).detected
    baseline_holds = not observe_fault(broken_root, (BASELINE,)).detected
    return healthy, probe_fails, baseline_holds


def classify_holdout_outcome(outcome: FixtureOutcome) -> str:
    """Why a hold-out did or did not recover -- the useful part of the score."""
    if outcome.recovered:
        return "recovered"
    if not outcome.detected:
        return "undetected"
    if outcome.diagnosis.fault_class is FaultClass.UNKNOWN:
        return "no_applicable_detector"
    if outcome.changed_files:
        return "wrong_repair_applied"
    return "misdiagnosed_then_rejected_by_regression_gate"


@dataclass(frozen=True)
class HoldoutComparison:
    known: BenchmarkResult
    holdout: BenchmarkResult
    breakdown: tuple[tuple[str, str], ...]

    @property
    def generalization_gap(self) -> float:
        """How much of the repair ability is pattern matching on known shapes."""
        return round(self.known.metrics.repair_success_rate - self.holdout.metrics.repair_success_rate, 3)


def run_holdout_benchmark(workspace: Path, fixtures: tuple[HoldoutFixture, ...] = HOLDOUT_FIXTURES,
                          verifier: RegressionVerifier | None = None) -> BenchmarkResult:
    outcomes = tuple(run_fixture(Path(workspace) / fixture.id, fixture.as_fault_fixture(), verifier)
                     for fixture in fixtures)
    control = HOLDOUT_CONTROL.as_fault_fixture(healthy=True)
    outcomes += (run_fixture(Path(workspace) / "H0", FaultFixture("H0", f"{control.intent}_CONTROL", control.symptom,
                                                                 FaultClass.HEALTHY, control.files,
                                                                 ExpectedAction.IGNORE), verifier),)
    notes = (
        "hold-out fixture は正しく動くモジュールに 1 箇所の改変を加えて生成しており、正解実装が常に存在します。",
        "診断器は hold-out を書く前に凍結しました。hold-out 名も正解 fault type も渡していません。",
        "expected_fault=UNKNOWN は『L8.6 の分類体系に該当項目が無い』という意味で、diagnosis_accuracy は『適用可能な検出器が無いことを正しく認識できた割合』として読みます。",
        "ここで得た失敗を診断器に取り込むと、この 6 件は既知故障になります。再測定には新しい hold-out が必要です。",
    )
    return BenchmarkResult(outcomes, summarize(outcomes), notes)


def compare_known_and_holdout(workspace: Path, verifier: RegressionVerifier | None = None) -> HoldoutComparison:
    workspace = Path(workspace)
    known = run_self_modification_benchmark(workspace / "known", FAULT_FIXTURES, verifier)
    holdout = run_holdout_benchmark(workspace / "holdout", HOLDOUT_FIXTURES, verifier)
    breakdown = tuple((item.fixture_id, classify_holdout_outcome(item)) for item in holdout.outcomes)
    return HoldoutComparison(known, holdout, breakdown)


def format_holdout_comparison(comparison: HoldoutComparison) -> str:
    known = comparison.known.metrics
    holdout = comparison.holdout.metrics
    lines = ["## Hold-out Self-Repair (L8.6.1)", "", "【既知故障 vs 未知故障】", "",
             "| 指標 | known (F1-F8) | hold-out (H1-H6) |",
             "| --- | --- | --- |",
             f"| repair_success_rate | `{known.repair_success_rate}` | `{holdout.repair_success_rate}` |",
             f"| fault_detection_rate | `{known.fault_detection_rate}` | `{holdout.fault_detection_rate}` |",
             f"| semantic_compile_rate | `{known.semantic_compile_rate}` | `{holdout.semantic_compile_rate}` |",
             f"| behavioral_decision_accuracy | `{known.behavioral_decision_accuracy}` | `{holdout.behavioral_decision_accuracy}` |",
             f"| false_repair_count | `{known.false_repair_count}` | `{holdout.false_repair_count}` |",
             f"| regression_introduction_rate | `{known.regression_introduction_rate}` | `{holdout.regression_introduction_rate}` |",
             "",
             f"**generalization_gap (repair_success): `{comparison.generalization_gap}`**",
             "", "【hold-out 内訳】", ""]
    lines.extend(f"- {identifier}: `{reason}`" for identifier, reason in comparison.breakdown)
    lines.extend(["", "【解釈上の注意】", *[f"- {item}" for item in comparison.holdout.notes]])
    return "\n".join(lines)


def main() -> None:  # pragma: no cover - manual entry point
    import tempfile

    with tempfile.TemporaryDirectory(prefix="l861-holdout-") as directory:
        comparison = compare_known_and_holdout(Path(directory))
        print(format_holdout_comparison(comparison))
        print()
        print(format_self_modification_report(comparison.holdout, "Hold-out detail (L8.6.1)"))


if __name__ == "__main__":  # pragma: no cover
    main()
