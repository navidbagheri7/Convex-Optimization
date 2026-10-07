"""Recover private TSPLIB cache files from the included verified numeric witnesses.

Use this offline path when the original TSPLIB host is unavailable. It does not
load or write a published optimal tour; distances are verified after roundtrip.
"""

import argparse
from pathlib import Path

import numpy as np

from optimization_lab.artifacts import sha256, write_json
from optimization_lab.routing import load_tsplib


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True)
    args = parser.parse_args()
    root = Path(args.data_root) / "optimization/routing"
    root.mkdir(parents=True, exist_ok=True)
    provenance = {}
    for name in ["berlin52", "ch150"]:
        source = Path("assets/source_data/routing") / f"{name}.npz"
        data = np.load(source)
        xy = data["coordinates"]
        path = root / f"{name}.tsp.gz"
        if path.exists():
            actual, _, distance = load_tsplib(path)
            assert actual == name
            np.testing.assert_array_equal(distance, data["distance"])
            origin = "Existing verified instance; not overwritten"
        else:
            text = f"NAME: {name}\nTYPE: TSP\nCOMMENT: Recovered from attributed TSPLIB numeric witness\nDIMENSION: {len(xy)}\nEDGE_WEIGHT_TYPE: EUC_2D\nNODE_COORD_SECTION\n"
            text += "".join(f"{i + 1} {x:.17g} {y:.17g}\n" for i, (x, y) in enumerate(xy)) + "EOF\n"
            path.write_text(text)
            actual, _, distance = load_tsplib(path)
            assert actual == name
            np.testing.assert_array_equal(distance, data["distance"])
            origin = (
                "Reconstructed from included numeric TSPLIB witness; plain text detected by bytes"
            )
        provenance[name] = dict(
            origin=origin,
            cache_sha256=sha256(path),
            numeric_witness_sha256=sha256(source),
            source="https://comopt.ifi.uni-heidelberg.de/software/TSPLIB95/",
        )
    write_json(root / "reconstruction_provenance.json", provenance)
    print(root)


if __name__ == "__main__":
    main()
