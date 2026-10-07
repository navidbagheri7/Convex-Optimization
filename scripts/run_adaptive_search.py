"""Run a declared random or sequential TPE MovieLens search."""

import argparse
import json
from pathlib import Path

from optimization_lab.adaptive_search import fit

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--config", default="configs/adaptive_search.json")
parser.add_argument("--seed", type=int, choices=[0, 1, 2], required=True)
parser.add_argument("--strategy", choices=["random", "tpe"], required=True)
parser.add_argument("--device", default="cuda:0")
parser.add_argument("--quick", action="store_true")
parser.add_argument("--output", help="Optional fresh output root; partial searches are preserved")
args = parser.parse_args()
config = json.loads(Path(args.config).read_text())
if args.quick:
    config |= dict(trials=2, startup_trials=1, epochs=2, rank_choices=[8])
fit(
    config,
    args.seed,
    args.strategy,
    args.device,
    args.output
    or ("results/smoke/adaptive_search" if args.quick else "results/reference/adaptive_search"),
    args.quick,
)
