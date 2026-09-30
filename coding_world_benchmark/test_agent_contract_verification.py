"""Tests for the launcher's contract check.

The check itself is the thing that says whether the running agent still has the
functions it is meant to have, so it needs to be able to fail: a verifier that
passes whatever it is given verifies nothing.
"""
from __future__ import annotations

import pytest

from coding_world_benchmark.agent_contract_verification import (
    INVARIANTS, STATE_KEYS, UNKNOWN_WORD_REQUEST, format_report, run,
    ui_api_calls, verify_bundle,
)


@pytest.fixture(scope="module")
def report():
    return run()


# -- what the UI asks for --------------------------------------------------

def test_the_endpoints_are_read_out_of_the_ui_source():
    found = ui_api_calls()
    assert "chat" in found and "health" in found and "features" in found
    # The panel's buttons pass action names to semanticAction; each one has to
    # correspond to an endpoint, which is what the parse is for.
    for action in ("answer", "open", "episode", "model", "reset", "state"):
        assert f"semantic/{action}" in found


def test_every_endpoint_the_ui_calls_is_exercised(report):
    check = next(item for item in report["checks"]
                 if item["name"] == "every endpoint the UI calls was exercised")
    assert check["ok"], check["detail"]


# -- the behaviour behind the endpoints ------------------------------------

def test_the_bridge_starts_the_way_electron_starts_it(report):
    assert report["spawned"]
    assert next(item for item in report["checks"]
                if item["name"] == "bridge is listening")["ok"]


def test_the_question_and_answer_loop_survives_the_http_boundary(report):
    names = {item["name"]: item for item in report["checks"]}
    for name in ("POST /api/semantic/ask",
                 "goal suspended, not restarted",
                 "「分からない」 commits nothing",
                 "POST /api/semantic/answer resumes the goal",
                 "a pending question owns the next utterance"):
        assert names[name]["ok"], names[name]["detail"]


def test_the_panel_gets_every_key_it_renders(report):
    names = {item["name"]: item for item in report["checks"]}
    assert names["GET /api/semantic/state"]["ok"]
    assert names["invariants reported"]["ok"]
    assert len(STATE_KEYS) >= 15 and len(INVARIANTS) == 4


def test_the_workspace_half_of_the_agent_is_untouched(report):
    names = {item["name"]: item for item in report["checks"]}
    assert names["POST /api/workspace"]["ok"]
    assert names["the workspace commands are still advertised"]["ok"]


def test_every_template_the_ui_offers_is_replayed(report):
    """The catalogue claims each form's behaviour; the run has to confirm it."""
    from coding_world_benchmark.semantic_service import TEMPLATES

    replayed = {item["name"] for item in report["checks"]
                if item["name"].startswith("template ")}
    assert replayed == {f"template {item.id}" for item in TEMPLATES}
    for item in report["checks"]:
        if item["name"].startswith("template "):
            assert item["ok"], f"{item['name']}: {item['detail']}"


def test_the_replay_checks_both_the_verdict_and_whether_it_committed(report):
    detail = next(item["detail"] for item in report["checks"]
                  if item["name"] == "template unknown")
    assert "NO_COMMIT" in detail and "committed=False" in detail


def test_the_code_form_of_every_template_lands_where_the_prose_form_does(report):
    """Two ways of writing the same move, verified to be the same move."""
    coded = [item for item in report["checks"]
             if item["name"].startswith("code form of")]
    assert len(coded) >= 8
    for item in coded:
        assert item["ok"], f"{item['name']}: {item['detail']}"


def test_a_script_is_refused_as_a_whole_over_http(report):
    check = next(item for item in report["checks"]
                 if item["name"] == "a script is refused as a whole")
    assert check["ok"], check["detail"]


def test_arithmetic_is_answered_and_nothing_else_is(report):
    names = {item["name"]: item for item in report["checks"]}
    for name in ("chat answers 1+1=?", "chat answers 2^10=?",
                 "the calculator refuses what is not arithmetic",
                 "prose with a number in it is not a calculation"):
        assert names[name]["ok"], names[name]["detail"]


def test_the_physics_script_refuses_to_guess_a_symbol(report):
    check = next(item for item in report["checks"]
                 if item["name"] == "an unbound symbol stops the physics script")
    assert check["ok"], check["detail"]


def test_the_trajectory_is_drawn_and_the_solver_is_checked(report):
    names = {item["name"]: item for item in report["checks"]}
    assert names["POST /api/semantic/script solves and draws"]["ok"]
    assert names["the solver is compared with the closed form"]["ok"]
    assert names["a misspelled keyword is refused with the nearest one"]["ok"]


def test_an_ambiguous_operator_is_carried_rather_than_chosen(report):
    names = {item["name"]: item for item in report["checks"]}
    for name in ("both readings of an ambiguous operator are given",
                 "the reader is told how to fix the reading",
                 "fixing the reading leaves one answer",
                 "a non-consequence is denied rather than assumed"):
        assert names[name]["ok"], names[name]["detail"]


def test_the_theorem_bundle_is_exercised_end_to_end(report):
    names = {item["name"]: item for item in report["checks"]}
    for name in ("GET /api/theorem/status", "POST /api/theorem/search",
                 "POST /api/theorem/run", "POST /api/theorem/obligations",
                 "POST /api/theorem/verify", "chat reaches the bundle"):
        assert names[name]["ok"], names[name]["detail"]


def test_a_sieved_candidate_is_not_sold_as_a_theorem(report):
    names = {item["name"]: item for item in report["checks"]}
    assert names["a sieved candidate is not reported as proved"]["ok"]
    assert names["a run without Lean says nothing was proved"]["ok"]


# -- the bundle the launcher builds ----------------------------------------

def test_the_bundle_check_looks_at_content_and_not_only_at_time():
    checks = {check.name: check for check in verify_bundle()}
    assert "bundle contains the Semantics panel" in checks
    assert "bundle carries the panel styles" in checks
    # A rebuild that dropped the panel would pass a timestamp check alone.
    assert checks["bundle contains the Semantics panel"].ok


def test_a_stale_bundle_is_reported_rather_than_ignored():
    """Editing the source after a build has to show up as a failed check."""
    checks = {check.name: check for check in verify_bundle()}
    staleness = checks["bundle is newer than its source"]
    assert staleness.detail.startswith("built ")


# -- the report ------------------------------------------------------------

def test_the_report_names_what_failed(report):
    text = format_report(report)
    assert "Coding agent contract verification" in text
    failed = [item for item in report["checks"] if not item["ok"]]
    if failed:
        assert "失敗した項目" in text
        for item in failed:
            assert item["name"] in text


def test_the_request_it_probes_with_is_the_one_with_an_unknown_word():
    assert "フガ率" in UNKNOWN_WORD_REQUEST
