import itertools

import numpy as np

from optimization_lab.routing import (
    Budget,
    cycle_crossover,
    held_karp,
    integer_one_tree,
    mutate,
    one_tree_bound,
    order_crossover,
    search,
    tour_length,
    two_opt,
)


def distances():
    xy = np.array([[0, 0], [2, 0], [2, 2], [0, 2], [1, 1]])
    return np.floor(np.linalg.norm(xy[:, None] - xy[None], axis=2) + 0.5).astype(np.int64)


def test_exact_dp_and_integer_bound_against_all_tours():
    d = distances()
    optimum = min(
        tour_length(d, np.array([0] + list(p))) for p in itertools.permutations(range(1, 5))
    )
    assert held_karp(d)["length"] == optimum
    for pi in [np.zeros(5, int), np.array([-20, 5, 30, -40, 7])]:
        result = integer_one_tree(d, pi)
        assert result["bound"] <= optimum
        assert result["degree"].sum() == 10
        assert len(result["edges"]) == 5
    result = one_tree_bound(d, optimum, 200)
    assert result["bound"] == optimum


def test_crossover_mutation_permutations():
    rng = np.random.default_rng(0)
    for n in [5, 20, 52]:
        for _ in range(30):
            a, b = rng.permutation(n), rng.permutation(n)
            for child in [
                order_crossover(a, b, rng),
                cycle_crossover(a, b),
                *(mutate(a, rng, method) for method in ["swap", "reverse", "insert"]),
            ]:
                np.testing.assert_array_equal(np.sort(child), np.arange(n))


def test_search_budget_and_two_opt_monotonicity():
    rng = np.random.default_rng(4)
    xy = rng.normal(size=(10, 2)) * 100
    d = np.floor(np.linalg.norm(xy[:, None] - xy[None], axis=2) + 0.5).astype(np.int64)
    for method in [
        "nn_2opt",
        "sa_swap",
        "sa_reverse",
        "sa_insert",
        "ga_ox_swap",
        "ga_cx_reverse",
        "as",
        "acs",
        "acs_2opt",
    ]:
        result = search(d, method, 0, 10000)
        assert result["edge_work"] <= 10000
        assert result["length"] == tour_length(d, result["tour"])
        assert np.all(np.diff(result["history"][:, 1]) < 0)
    budget = Budget(d, 10000)
    start = np.arange(10)
    assert tour_length(d, two_opt(budget, start)) <= tour_length(d, start)


def test_certificate_overflow_guard_handles_minimum_signed_integer():
    import pytest

    with pytest.raises(ValueError):
        integer_one_tree(distances(), np.full(5, np.iinfo(np.int64).min, dtype=np.int64))
