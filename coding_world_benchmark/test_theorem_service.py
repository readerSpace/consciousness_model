"""Tests for the theorem-discovery bundle seen from the agent.

The claim under test is not that the bundle works -- it has its own suite for
that, and this module can run it. It is that the agent reports what the bundle
actually did, and in particular never turns a sieved candidate into a theorem.
"""
from __future__ import annotations

import pytest

from coding_world_benchmark import theorem_service
from coding_world_benchmark.theorem_service import (
    DEFAULT_CYCLES, format_run, format_search, handle_command, run_discovery,
    search, status, verify,
)


@pytest.fixture(scope="module")
def run():
    outcome = run_discovery(12)
    assert outcome.ok, outcome.detail
    return outcome.payload


# -- the bundle is there ---------------------------------------------------

def test_the_bundle_is_found_beside_the_agent():
    found = status()
    assert found["available"]
    assert found["path"].endswith("theorem_discovery_system")


def test_the_corpus_is_read_directly():
    found = status()
    assert found["corpus"] == 42
    assert set(found["domains"]) >= {"A", "B", "dv"}


# -- searching -------------------------------------------------------------

def test_search_matches_names_domains_premises_and_statements():
    names = [item["name"] for item in search("assoc")]
    assert {"dA_add_assoc", "dA_mul_assoc", "dB_inter_assoc"} <= set(names)
    assert [item["name"] for item in search("Set.inter_assoc")] == ["dB_inter_assoc"]


def test_search_respects_its_limit():
    assert len(search("", limit=3)) == 3


def test_a_miss_says_so_rather_than_inventing_one():
    spoken = format_search("frobnicate", search("frobnicate"))
    assert "ありません" in spoken


# -- running ---------------------------------------------------------------

def test_a_run_reports_the_cycles_it_was_asked_for(run):
    assert run["summary"]["cycles"] == 12
    assert run["pool"] > 0


def test_every_asserted_candidate_carries_where_it_came_from(run):
    for item in run["asserted"]:
        assert item["name"] and item["mode"] and item["source"]
        assert 0.0 <= item["confidence"] <= 1.0
        assert isinstance(item["known"], bool)


def test_cycles_are_bounded():
    assert run_discovery(10 ** 9).payload["summary"]["cycles"] \
        <= theorem_service.MAX_CYCLES


def test_the_same_seed_gives_the_same_run():
    first = run_discovery(8).payload["summary"]
    second = run_discovery(8).payload["summary"]
    assert first["asserted"] == second["asserted"]
    assert first["models_tested"] == second["models_tested"]


# -- the distinction that must not collapse --------------------------------

def test_asserted_and_valid_are_reported_separately(run):
    spoken = format_run(run)
    assert "asserted（篩を通った）" in spoken and "valid（証明された）" in spoken


def test_a_sieved_candidate_is_never_called_proved(run):
    spoken = format_run(run)
    assert "「証明された」ではありません" in spoken


def test_a_run_without_lean_says_nothing_was_proved(run):
    summary = run["summary"]
    spoken = format_run(run)
    if summary["lean_calls"] == 0:
        assert "1 回も呼ばれていません" in spoken
        assert summary["valid"] == 0
    else:
        assert "証明済み" in spoken


def test_the_obligations_are_a_lean_file_of_what_still_needs_proving(run):
    assert run["obligations"].startswith("import Mathlib")


# -- the bundle's own suite ------------------------------------------------

def test_the_bundle_verifies_itself():
    """Also what says the transplanted corpus path is right."""
    outcome = verify()
    assert outcome.ok, outcome.detail
    assert outcome.payload["counts"].get("passed", 0) >= 50
    assert outcome.payload["counts"].get("failed", 0) == 0


# -- commands --------------------------------------------------------------

@pytest.mark.parametrize("message, wanted", [
    ("/theorem", "定理発見システム"),
    ("/theorem search assoc", "dA_add_assoc"),
    ("/theorem search", "使い方"),
])
def test_commands_answer(message, wanted):
    spoken = handle_command(message)
    assert spoken is not None and wanted in spoken


@pytest.mark.parametrize("message", [
    "こんにちは", "/inspect", "1+1=?", "state()", "/semantics",
])
def test_everything_else_is_left_alone(message):
    assert handle_command(message) is None


def test_the_status_names_the_sieve_and_the_prover_apart():
    spoken = handle_command("/theorem")
    assert "篩（asserted）と証明（valid）は別のもの" in spoken


# -- when the bundle is not there -----------------------------------------

def test_a_missing_bundle_is_reported_rather_than_raised(monkeypatch, tmp_path):
    monkeypatch.setattr(theorem_service, "BUNDLE", tmp_path / "nowhere")
    monkeypatch.setattr(theorem_service, "CORPUS", tmp_path / "nowhere.json")
    assert not theorem_service.available()
    assert theorem_service.search("assoc") == []
    outcome = theorem_service.run_discovery(4)
    assert not outcome.ok and "見つかりません" in outcome.detail
    assert "見つかりません" in handle_command("/theorem")


def test_a_run_that_overruns_is_cut_off_rather_than_left_hanging():
    """The bundle's notes record an experiment that does not stop."""
    outcome = theorem_service.run_discovery(400, timeout=0)
    assert not outcome.ok
    assert "打ち切" in outcome.detail or "起動できません" in outcome.detail
