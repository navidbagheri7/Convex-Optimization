"""Recompute every selected model's test aggregates from fresh portable checkpoints."""

import argparse
import json
from pathlib import Path

import numpy as np

from optimization_lab.artifacts import sha256, write_json
from optimization_lab.deployment import load_factors, predict_ratings
from optimization_lab.recommendation import load_ratings, objective, predict, score


def gradient(state, data, lam, beta):
    fit = data["fit"]
    users, items = fit[:, 0], fit[:, 1]
    residual = predict(state, users, items) - fit[:, 2]
    gu = lam * data["user_degree"][:, None] * state["user"]
    gi = lam * data["item_degree"][:, None] * state["item"]
    np.add.at(gu, users, residual[:, None] * state["item"][items])
    np.add.at(gi, items, residual[:, None] * state["user"][users])
    gb = (
        np.bincount(users, weights=residual, minlength=data["users"])
        + beta * data["user_degree"] * state["user_bias"]
    )
    gc = (
        np.bincount(items, weights=residual, minlength=data["items"])
        + beta * data["item_degree"] * state["item_bias"]
    )
    values = np.concatenate([gu.ravel(), gi.ravel(), gb, gc]) / len(fit)
    return dict(
        gradient_infinity_norm=float(abs(values).max()),
        gradient_l2_norm=float(np.linalg.norm(values)),
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True)
    args = parser.parse_args()
    rows = []
    for family in ["recommendation_fixed", "recommendation"]:
        for seed in [0, 1, 2]:
            root = Path("results/reference") / family / f"seed{seed}"
            record = json.loads((root / "run.json").read_text())
            metrics = json.loads((root / "metrics.json").read_text())
            cfg = record["config"]
            data = load_ratings(args.data_root, cfg["validation_fraction"], cfg["split_seed"])
            assert data["source_sha256"] == cfg["source_data_sha256"]
            for row in metrics["rows"]:
                path = root / "checkpoints" / f"{row['method']}.npz"
                assert sha256(path) == row["checkpoint_sha256"]
                state = load_factors(path)
                calculated = score(state, data["test"])
                deployed = predict_ratings(path, data["test"][:, 0], data["test"][:, 1])
                np.testing.assert_array_equal(deployed, calculated["prediction"])
                np.testing.assert_allclose(
                    calculated["rmse"], row["test"]["rmse"], atol=1e-12, rtol=0
                )
                errors = calculated["prediction"] - data["test"][:, 2]
                evidence = np.load(
                    Path("assets/source_data") / family / f"{row['method']}_{seed}_test.npz"
                )
                for key, weights in [
                    ("user_rating_count", None),
                    ("user_squared_error", errors**2),
                    ("user_absolute_error", abs(errors)),
                ]:
                    actual = np.bincount(
                        data["test"][:, 0], weights=weights, minlength=data["users"]
                    )
                    np.testing.assert_allclose(actual, evidence[key], atol=1e-10, rtol=0)
                penalty = row.get("selected_lambda", cfg["lambda"])
                np.testing.assert_allclose(
                    objective(state, data, penalty, cfg["bias_penalty"]),
                    row["selected_training_objective"],
                    atol=1e-12,
                    rtol=0,
                )
                rows.append(
                    dict(
                        family=family,
                        seed=seed,
                        method=row["method"],
                        checkpoint_sha256=sha256(path),
                        test_rmse=calculated["rmse"],
                        **gradient(state, data, penalty, cfg["bias_penalty"]),
                    )
                )
    assert len(rows) == 30
    write_json(
        "docs/checkpoint_audit.json",
        dict(
            status="passed",
            selected_models=len(rows),
            rows=rows,
            scope="Fresh NumPy float64 predictions and independently recomputed per-user error aggregates. Gradient diagnostics use the declared normalized objective; validation-selected models are not claimed stationary or globally optimal. Deterministic baselines constrain factor rank0; the global-mean baseline is not an optimized bias model.",
        ),
    )
    print("Audited30 selected models")


if __name__ == "__main__":
    main()
