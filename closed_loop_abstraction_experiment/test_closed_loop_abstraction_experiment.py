import pytest

from closed_loop_abstraction_experiment import (
    KNOWN_CLOSED_PATH,
    Config,
    closed_paths,
    compressed_concepts,
    displacement,
    invariance_metrics,
    path_name,
    select_paths,
)


def test_composition_recovers_zero_displacement_path_from_primitives():
    assert displacement(KNOWN_CLOSED_PATH) == (0, 0)
    assert KNOWN_CLOSED_PATH in closed_paths()
    assert compressed_concepts() == ("closed_loop",)


def test_closed_loop_category_predicts_gauge_invariance():
    precision, recall = invariance_metrics()
    assert precision == pytest.approx(1.0)
    assert recall == pytest.approx(1.0)


def test_prediction_selection_retains_a_derived_closed_path():
    selected = select_paths(Config(train_steps=8))
    closed_names = {path_name(path) for path in closed_paths()}
    assert closed_names & set(selected)