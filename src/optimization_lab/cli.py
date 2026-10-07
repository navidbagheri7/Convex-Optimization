"""Reproducible studies with bundled default configurations."""

import argparse
import json
import os
from pathlib import Path

from .resources import config_path


def main():
    parser = argparse.ArgumentParser(
        description="Numerical optimization, certificates and comparisons"
    )
    parser.add_argument("--version", action="version", version="certified-optimization-lab 1.0.0")
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ["convex", "first-order", "spectral", "recommendation", "routing"]:
        command = commands.add_parser(name)
        command.add_argument("--config", type=Path)
        command.add_argument("--output", type=Path)
        command.add_argument(
            "--quick", action="store_true", help="Small smoke protocol; not a reference result"
        )
        if name == "recommendation":
            command.add_argument("--data-root", required=True)
            command.add_argument("--seed", type=int, default=0)
            command.add_argument("--device", default="cpu")
        if name == "routing":
            command.add_argument("--instance-root", required=True)
    commands.add_parser("report")
    commands.add_parser("validate")
    data = commands.add_parser("data")
    data.add_argument(
        "--data-root",
        default=os.environ.get(
            "OPTIMIZATION_DATA_ROOT", str(Path.home() / ".cache/optimization-lab")
        ),
    )
    data.add_argument("--dataset", choices=["movielens", "routing"], required=True)
    args = parser.parse_args()
    if args.command == "report":
        from .reporting import report

        return report()
    if args.command == "validate":
        from .validation import validate

        return validate()
    if args.command == "data":
        from .data import prepare

        return prepare(args.dataset, args.data_root)
    config = json.loads(
        (args.config or config_path(args.command.replace("-", "_") + ".json")).read_text()
    )
    output = args.output or Path(
        "results/smoke" if args.quick else "results/reference"
    ) / args.command.replace("-", "_")
    if args.command == "convex":
        from .experiments import convex

        return convex(config, output, args.quick)
    if args.command == "first-order":
        from .first_order_study import study

        if args.quick:
            for cfg in config.values():
                if isinstance(cfg, dict) and "steps" in cfg:
                    cfg["steps"] = 100
        return study(config, output, args.quick)
    if args.command == "spectral":
        from .spectral_study import study

        return study(config, output, args.quick)
    if args.command == "recommendation":
        from .recommendation import fit

        if args.quick:
            config |= dict(
                epochs=2, rank=8, lambdas=[0.1], learning_rates=dict(sgd=[1.0], adam=[0.003])
            )
        return fit(config, args.data_root, args.seed, args.device, args.quick, output)
    if args.command == "routing":
        from .routing_study import run

        if args.quick:
            config |= dict(edge_budget=50000, bound_iterations=100, ortools_seconds=1.0)
        return run(config, args.instance_root, output, args.quick)


if __name__ == "__main__":
    main()
