"""Routing comparison, including independently re-evaluable integer certificates."""

import argparse
import json
from pathlib import Path

import numpy as np

from .artifacts import Run, sha256, write_json
from .routing import held_karp, load_tsplib, one_tree_bound, ortools_search, search


def run(config, instance_root, output="results/reference/routing", quick=False):
    instance_root = Path(instance_root)
    destination = Path(output) / "source_data" if quick else Path("assets/source_data/routing")
    destination.mkdir(parents=True, exist_ok=True)
    instances = config["instances"][:1] if quick else config["instances"]
    for name in instances:
        path = instance_root / f"{name}.tsp.gz"
        actual, xy, distance = load_tsplib(path)
        if actual != name:
            raise ValueError("Instance name disagrees with declared protocol")
        optimum = {"berlin52": 7542, "ch150": 6528}[name]
        run = Run(
            Path(output) / name,
            config | dict(instance=name, quick=quick, instance_sha256=sha256(path)),
            [__file__, Path(__file__).with_name("routing.py")],
        )
        arrays = {"distance": distance, "coordinates": xy}
        rows = []
        seeds = config["seeds"][:1] if quick else config["seeds"]
        methods = config["methods"][:2] if quick else config["methods"]
        for seed in seeds:
            for method in methods + ["ortools_gls"]:
                result = (
                    ortools_search(distance, seed % len(distance), config["ortools_seconds"])
                    if method == "ortools_gls"
                    else search(distance, method, seed, config["edge_budget"])
                )
                arrays[f"{method}_seed{seed}_tour"] = result.pop("tour")
                if "history" in result:
                    arrays[f"{method}_seed{seed}_history"] = result.pop("history")
                result |= dict(
                    method=method,
                    seed=seed,
                    gap_to_published_optimum_percent=100 * (result["length"] / optimum - 1),
                )
                rows.append(result)
                print(name, seed, method, result["length"], flush=True)
        # The bound's step sizes use our first computed incumbent, never the published optimum.
        upper = rows[0]["length"]
        certificate = one_tree_bound(distance, upper, config["bound_iterations"])
        for key in ["potentials", "degree", "edges", "history"]:
            arrays[f"one_tree_{key}"] = certificate.pop(key)
        np.savez_compressed(destination / f"{name}.npz", **arrays)
        run.finish(
            dict(
                rows=rows,
                one_tree_certificate=certificate,
                bound_incumbent=upper,
                published_optimum=optimum,
                optimum_reference="https://comopt.ifi.uni-heidelberg.de/software/TSPLIB95/tsp/TSP-BEST.html",
                protocol="EUC_2D nearest-integer distances and cyclic return. Owned methods cap counted distance-edge work, including initialization, construction weights, objective evaluations and 2opt checks. Crossover, sampling, probability arithmetic and dense matrix updates are not counted; elapsed time is also reported. NN+2opt stops at local optimum. AS deposits all-ant trails; ACS includes local construction updates and best-tour global updates; ACS+2opt consumes the same budget. OR-Tools uses a separate5s wall-clock limit and callback counts, not a matched edge budget. Seed changes its depot, not a claimed solver random seed; timing can vary. Exact integer1-tree bound uses rounded integer potentials, with float proposals only. No official optimal tour loaded.",
            )
        )
    small = distance[:12, :12]
    exact = held_karp(small)
    bound = one_tree_bound(small, exact["length"], config["bound_iterations"])
    arrays = dict(
        distance=small, tour=exact.pop("tour"), potentials=bound["potentials"], edges=bound["edges"]
    )
    np.savez_compressed(destination / "exact12.npz", **arrays)
    write_json(
        Path(output) / "exact12.json",
        dict(
            exact=exact,
            lower_bound=bound["bound"],
            scope="First12 nodes of the last processed instance; Held-Karp gives an exact optimum.",
        ),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance-root", default="legacy/originals")
    parser.add_argument("--config", default="configs/routing.json")
    parser.add_argument("--output", default="results/reference/routing")
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text())
    if args.quick:
        config |= dict(edge_budget=50000, bound_iterations=100, ortools_seconds=0.2)
    run(config, args.instance_root, args.output, args.quick)
