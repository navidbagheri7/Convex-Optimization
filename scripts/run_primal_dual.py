"""Run owned TV solver comparisons with explicit reference/smoke separation."""

import argparse
import json
from pathlib import Path

from optimization_lab.primal_dual_study import study

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--config", default="configs/primal_dual.json")
parser.add_argument("--quick", action="store_true")
args = parser.parse_args()
config = json.loads(Path(args.config).read_text())
if args.quick:
    config |= dict(
        seeds=[0], datasets=["piecewise"], size=16, oracle_size=8, lambdas=[0.06], steps=800
    )
study(
    config,
    "results/smoke/primal_dual" if args.quick else "results/reference/primal_dual",
    args.quick,
)
