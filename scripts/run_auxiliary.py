"""Regenerate transformed-domain and exact-small-TSP evidence."""

from pathlib import Path

import numpy as np

from optimization_lab.artifacts import Run
from optimization_lab.exact import mtz_tsp
from optimization_lab.geometry import transformed_domains
from optimization_lab.routing import held_karp, integer_one_tree


def main():
    run = Run(
        "results/reference/geometry",
        {"scope": "four supplied transformed convex forms"},
        ["src/optimization_lab/geometry.py"],
    )
    run.finish(transformed_domains())
    # The public numeric witness removes dependence on private course files.
    with np.load("assets/source_data/exact/tsp12.npz") as data:
        distance = data["distance"].copy()
        potentials = data["bound_potentials"].copy()
    run = Run(
        "results/reference/exact",
        {"instance": "ch150 first12 nodes", "seconds": 30},
        ["src/optimization_lab/exact.py", "src/optimization_lab/routing.py"],
    )
    dp = held_karp(distance)
    milp = mtz_tsp(distance)
    bound = integer_one_tree(distance, potentials)
    assert dp["length"] == milp["length"] and milp["proven_optimal"]
    assert bound["bound"] <= dp["length"]
    Path("assets/source_data/exact").mkdir(exist_ok=True)
    np.savez_compressed(
        "assets/source_data/exact/tsp12.npz",
        distance=distance,
        dp_tour=dp.pop("tour"),
        milp_tour=milp.pop("tour"),
        bound_potentials=potentials,
        bound_edges=bound["edges"],
    )
    run.finish(dict(dynamic_programming=dp, milp=milp, integer_one_tree_bound=bound["bound"]))


if __name__ == "__main__":
    main()
