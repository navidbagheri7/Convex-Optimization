"""Complete-budget convex application studies and exact numerical provenance."""

import json
from pathlib import Path

import numpy as np

from . import convex_cases
from .artifacts import Run, sha256
from .resources import config_path


def convex(config, output="results/reference/convex", quick=False):
    sources = [
        __file__,
        Path(__file__).with_name("convex_cases.py"),
        Path(__file__).with_name("barrier.py"),
    ]
    cases = config["cases"] if not quick else ["revenue", "blend", "psd"]
    for case in cases:
        if case in ["revenue"]:
            path = config_path("additional_parameters.json")
            parameters = json.loads(path.read_text())[case]
        elif case == "psd":
            path = None
            parameters = {"matrix": [[1.0, -1.0, 0.0], [-1.0, 0.8, -1.0], [0.0, -1.0, 100.0]]}
        else:
            path = config_path(f"{case}_parameters.json")
            parameters = json.loads(path.read_text())
        run = Run(
            Path(output) / case,
            config
            | {
                "case": case,
                "parameters": parameters,
                "quick": quick,
                "parameter_sha256": None if path is None else sha256(path),
            },
            sources,
        )
        functions = {
            "speed": convex_cases.speed,
            "revenue": convex_cases.revenue,
            "blend": convex_cases.geometric_blend,
            "psd": lambda p: convex_cases.nearest_psd(p["matrix"]),
            "regulator": convex_cases.regulator,
            "evacuation": convex_cases.evacuation,
            "portfolio": convex_cases.portfolio,
        }
        result = functions[case](parameters)
        arrays = result.pop("arrays")
        destination = run.directory / "source_data" if quick else Path("assets/source_data/convex")
        destination.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(destination / f"{case}.npz", **arrays)
        run.finish(result)
        print("Complete convex application", case, flush=True)
