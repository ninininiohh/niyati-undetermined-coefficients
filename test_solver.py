import sympy as sp
import pytest

from solver import operator, solve_config, x


@pytest.mark.parametrize(
    "config",
    [
        {"family": "Exponential", "k": 1, "a": 3},
        {"family": "Sine", "k": 1, "a": 2, "b": 0},
        {"family": "Polynomial", "k": 1, "degree": 2},
        {"family": "Exponential × sine", "k": 1, "c": 3, "a": 2, "b": 0},
        {"family": "Exponential", "k": 1, "a": -1},
        {"family": "Exponential × polynomial", "k": 2, "a": 3, "degree": 2},
        {"family": "Polynomial × cosine", "k": 3, "degree": 2, "a": 2, "b": 1},
    ],
)
def test_particular_solution_has_zero_residual(config):
    result = solve_config(config)
    assert sp.simplify(operator(result.particular) - result.forcing) == 0


def test_exponential_resonance_multiplies_trial_by_x():
    result = solve_config({"family": "Exponential", "k": 1, "a": -1})
    assert result.resonance_power == 1
    assert sp.simplify(result.particular - x * sp.exp(-x)) == 0


def test_zero_frequency_is_reduced_without_a_dependent_trial():
    result = solve_config({"family": "Sine", "k": 2, "a": 0, "b": 1})
    assert sp.simplify(operator(result.particular) - result.forcing) == 0
