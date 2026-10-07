"""Explicit genetic/local-search hybrids, kept separate from textbook GA recipes."""

import argparse
import json
import time
from pathlib import Path

import numpy as np

from .artifacts import Run
from .routing import (
    Budget,
    cycle_crossover,
    load_tsplib,
    mutate,
    nearest_neighbor,
    order_crossover,
    two_opt,
)


def hybrid(distance, seed, edge_budget, crossover="ox", population_size=24):
    budget = Budget(distance, edge_budget)
    rng = np.random.default_rng(seed)
    population, costs = [], []
    n = len(distance)
    for _ in range(population_size):
        route = nearest_neighbor(budget, int(rng.integers(n)))
        route = mutate(route, rng, "reverse")
        budget.evaluate(route)
        route = two_opt(budget, route, max_passes=2)
        population.append(route)
        # Re-evaluation counted; no hidden objective evaluations.
        costs.append(budget.evaluate(route))
    cross = order_crossover if crossover == "ox" else cycle_crossover
    children = 0
    while budget.room(2 * n + 4):
        parents = []
        for _ in range(2):
            ids = rng.choice(population_size, 3, replace=False)
            parents.append(population[min(ids, key=lambda i: costs[i])])
        route = cross(*parents, rng)
        if rng.random() < 0.3:
            route = mutate(route, rng, "reverse")
        budget.evaluate(route)
        # Reserve one objective evaluation after local search.
        original_limit = budget.limit
        budget.limit -= n
        route = two_opt(budget, route, max_passes=2)
        budget.limit = original_limit
        value = budget.evaluate(route)
        worst = int(np.argmax(costs))
        if value <= costs[worst]:
            population[worst], costs[worst] = route, value
        children += 1
        if children % 20 == 0 and budget.room(2 * n + n * (n - 1) // 2):
            route = nearest_neighbor(budget, int(rng.integers(n)))
            route = mutate(route, rng, "reverse")
            population[worst], costs[worst] = route, budget.evaluate(route)
    return dict(
        tour=budget.tour,
        length=budget.best,
        history=np.array(budget.history),
        edge_work=budget.work,
        objective_evaluations=budget.objectives,
        children=children,
        elapsed_seconds=time.perf_counter() - budget.start,
    )


def study(config, instance_root, output="results/reference/memetic"):
    destination = Path("assets/source_data/memetic")
    destination.mkdir(parents=True, exist_ok=True)
    for name in config["instances"]:
        path = Path(instance_root) / (name + ".tsp.gz")
        actual, _, d = load_tsplib(path)
        assert actual == name
        run = Run(
            Path(output) / name,
            config | dict(instance=name),
            [__file__, Path(__file__).with_name("routing.py")],
        )
        arrays, rows = {}, []
        for seed in config["seeds"]:
            for cross in ["ox", "cx"]:
                result = hybrid(d, seed, config["edge_budget"], cross)
                method = f"ga_{cross}_reverse_2opt"
                arrays[f"{method}_seed{seed}_tour"] = result.pop("tour")
                arrays[f"{method}_seed{seed}_history"] = result.pop("history")
                result |= dict(
                    method=method,
                    seed=seed,
                    gap_to_published_optimum_percent=100
                    * (result["length"] / {"berlin52": 7542, "ch150": 6528}[name] - 1),
                )
                rows.append(result)
                print(name, seed, method, result["length"], flush=True)
        np.savez_compressed(destination / f"{name}.npz", **arrays)
        run.finish(
            dict(
                rows=rows,
                scope="Separate memetic GA extension:24 nearest-neighbor / perturbed / two-pass2opt initial members, tournament3, OX or CX, reversal mutation0.3, two2opt passes per child, worst replacement and explicit20-child immigrant. Same10M counted distance-edge budget including local improvement. This is not the pure original GA comparison.",
            )
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance-root", required=True)
    args = parser.parse_args()
    study(json.loads(Path("configs/routing.json").read_text()), args.instance_root)
