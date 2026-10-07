"""Verify all six selected search checkpoints against actual full-test predictions."""

import json
import subprocess
import sys
from pathlib import Path

import numpy as np

from optimization_lab.artifacts import sha256, write_json
from optimization_lab.deployment import load_factors, predict_ratings
from optimization_lab.recommendation import load_ratings, score


def main():
    config = json.loads(Path("configs/adaptive_search.json").read_text())
    data = load_ratings(config["data_root"], config["validation_fraction"], config["split_seed"])
    output = Path(".local/adaptive_inference")
    output.mkdir(exist_ok=True)
    inp = output / "pairs.csv"
    np.savetxt(
        inp, data["test"][:64, :2], fmt="%d", delimiter=",", header="user_id,item_id", comments=""
    )
    checks = []
    for strategy in config["strategies"]:
        for seed in config["seeds"]:
            root = Path("results/reference/adaptive_search") / strategy / f"seed{seed}"
            record = json.loads((root / "run.json").read_text())
            metrics = json.loads((root / "metrics.json").read_text())
            assert data["source_sha256"] == record["config"]["source_data_sha256"]
            path = root / "checkpoints/selected.npz"
            assert sha256(path) == metrics["selected_checkpoint_sha256"]
            raw = score(load_factors(path), data["test"])
            deployed = predict_ratings(path, data["test"][:, 0], data["test"][:, 1])
            np.testing.assert_array_equal(deployed, raw["prediction"])
            np.testing.assert_allclose(
                [raw["rmse"], raw["mae"]],
                [metrics["test"]["rmse"], metrics["test"]["mae"]],
                atol=1e-12,
                rtol=0,
            )
            errors = deployed - data["test"][:, 2]
            with np.load(
                f"assets/source_data/adaptive_search/{strategy}/seed{seed}_test.npz",
                allow_pickle=False,
            ) as saved:
                for key, weights in [
                    ("user_rating_count", None),
                    ("user_squared_error", errors**2),
                    ("user_absolute_error", abs(errors)),
                ]:
                    np.testing.assert_allclose(
                        np.bincount(data["test"][:, 0], weights=weights, minlength=data["users"]),
                        saved[key],
                        atol=1e-10,
                        rtol=0,
                    )
            hashes = []
            for attempt in [0, 1]:
                out = output / f"{strategy}_{seed}_{attempt}.csv"
                subprocess.run(
                    [
                        sys.executable,
                        "scripts/predict_ratings.py",
                        "--checkpoint",
                        str(path),
                        "--input",
                        str(inp),
                        "--output",
                        str(out),
                    ],
                    check=True,
                )
                predicted = np.loadtxt(out, delimiter=",", skiprows=1)[:, 2]
                np.testing.assert_array_equal(predicted, deployed[:64])
                sidecar = json.loads(out.with_suffix(".json").read_text())
                assert sidecar["input_sha256"] == sha256(inp) and sidecar[
                    "checkpoint_sha256"
                ] == sha256(path)
                hashes.append(sha256(out))
            assert hashes[0] == hashes[1]
            checks.append(
                dict(
                    strategy=strategy,
                    seed=seed,
                    test_ratings=20000,
                    checkpoint_sha256=sha256(path),
                    test_rmse=raw["rmse"],
                    all_predictions_bitwise_equal=True,
                    repeated_command_csv_sha256=hashes[0],
                )
            )
    write_json(
        "docs/adaptive_checkpoint_audit.json",
        dict(
            status="passed",
            selected_models=6,
            checks=checks,
            source_sha256={
                str(p): sha256(p)
                for p in [
                    Path("scripts/audit_adaptive_checkpoints.py"),
                    Path("scripts/predict_ratings.py"),
                    Path("src/optimization_lab/deployment.py"),
                    Path("src/optimization_lab/recommendation.py"),
                ]
            },
            scope="Every one of 20,000 official-test predictions per selected search model; full per-user errors and all selected scores recomputed. Two fresh standalone CSV commands per model reproduce portable float64 predictions exactly. Raw ID pairs and selected factors remain private.",
        ),
    )
    print("Verified all six selected search models and twelve fresh-process prediction commands")


if __name__ == "__main__":
    main()
