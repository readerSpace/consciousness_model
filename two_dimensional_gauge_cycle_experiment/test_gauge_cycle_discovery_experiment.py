import pytest

from gauge_cycle_discovery_experiment import CYCLE, Config, gauge_transform, generated_grammar, make_state, select_concepts, values


def test_grammar_generates_the_closed_four_link_cycle_without_plaquette_name():
    assert CYCLE in generated_grammar()


def test_closed_cycle_value_is_gauge_invariant():
    original = values(make_state(6, 1))[CYCLE]
    transformed = values(gauge_transform(make_state(6, 1), 2))[CYCLE]
    assert original == pytest.approx(transformed, abs=1e-12)


def test_predictive_selection_includes_the_closed_cycle():
    selected, errors = select_concepts(Config(train_steps=8))
    assert CYCLE in selected
    assert len(errors) == 3