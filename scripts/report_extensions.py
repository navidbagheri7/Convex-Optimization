"""Independently recompute TV certificates and adaptive-search selection/metrics."""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from optimization_lab.artifacts import sha256, write_json
from optimization_lab.primal_dual import certificate


def check_record(directory):
    """Reject incomplete records and altered scientific source."""
    record = json.loads((directory / "run.json").read_text())
    assert record["status"] == "complete" and not record["config"].get("quick", False)
    for name, digest in record["source_sha256"].items():
        assert sha256(Path("src/optimization_lab") / name) == digest
    return record, json.loads((directory / "metrics.json").read_text())


def report_tv():
    config = json.loads(Path("configs/primal_dual.json").read_text())
    rows = []
    for case in config["datasets"]:
        for seed in config["seeds"]:
            for lam in config["lambdas"]:
                name = f"{case}_seed{seed}_lambda{lam:g}"
                record, metrics = check_record(Path("results/reference/primal_dual") / name)
                assert all(record["config"][key] == value for key, value in config.items())
                assert (
                    record["config"]["dataset"],
                    record["config"]["seed"],
                    record["config"]["penalty"],
                ) == (case, seed, lam)
                assert sha256(metrics["evidence"]) == metrics["evidence_sha256"]
                with np.load(metrics["evidence"], allow_pickle=False) as data:
                    assert (
                        data["truth"].shape
                        == data["observed"].shape
                        == (config["size"], config["size"])
                    )
                    for row in metrics["rows"]:
                        method = row["method"]
                        cert = certificate(
                            data[method + "_x"], data["observed"], data[method + "_dual"], lam
                        )
                        for key, value in cert.items():
                            np.testing.assert_allclose(
                                value, row["certificate"][key], atol=1e-10, rtol=0
                            )
                        assert (
                            -1e-8
                            <= cert["gap"]
                            <= config["gap_per_pixel"] * data["truth"].size * 1.000001
                        )
                        assert row["converged"] and row["iterations"] <= config["steps"]
                        mse = float(np.mean((data[method + "_x"] - data["truth"]) ** 2))
                        np.testing.assert_allclose(mse, row["mse"], atol=1e-14, rtol=0)
                        np.testing.assert_allclose(
                            -10 * np.log10(max(mse, 1e-16)), row["psnr"], atol=1e-12, rtol=0
                        )
                        small = certificate(
                            data[method + "_oracle_x"],
                            data["oracle_observed"],
                            data[method + "_oracle_dual"],
                            lam,
                        )
                        np.testing.assert_allclose(
                            small["primal"] - metrics["oracle"]["objective"],
                            row["oracle_objective_difference"],
                            atol=1e-10,
                            rtol=0,
                        )
                        assert abs(row["oracle_objective_difference"]) < 1e-5
                        assert cert["dual_feasibility"] <= 1e-12
                        history = data[method + "_history"]
                        assert history[-1, 0] == row["iterations"] and history.shape[1] == 5
                        np.testing.assert_allclose(
                            history[-1, 1:4],
                            [cert["primal"], cert["dual"], cert["gap"]],
                            atol=1e-10,
                        )
                        np.testing.assert_array_equal(
                            data[method + "_snapshots"][-1], data[method + "_x"]
                        )
                        assert data[method + "_snapshot_iterations"][-1] == row["iterations"]
                        rows.append(
                            dict(
                                dataset=case,
                                seed=seed,
                                penalty=lam,
                                method=method,
                                psnr=row["psnr"],
                                gap_per_pixel=cert["gap"] / data["truth"].size,
                                iterations=row["iterations"],
                                elapsed_seconds=row["elapsed_seconds"],
                                oracle_objective_difference=row["oracle_objective_difference"],
                            )
                        )
    assert len(rows) == 72
    frame = pd.DataFrame(rows)
    frame.to_csv("docs/primal_dual_fits.csv", index=False)
    lines = [
        "# Primal–dual image reconstruction",
        "",
        "Minimize `0.5 ||x-b||² + λ Σpixel ||Dx||₂` with periodic forward differences. PDHG, strongly-convex accelerated PDHG, projected dual FISTA and FFT ADMM solve the same unconstrained float-image objective. A projected dual field gives a feasible Fenchel lower bound independently of solver. We stop at a gap per pixel of 1e-6, rather than comparing arbitrary iteration counts.",
        "",
        "All 72 solves cover three noise seeds, two images and three fixed penalties. The procedural image is identified separately from the attributed photograph. Penalties are diagnostic controls, not selected by test PSNR. Displays clip pixels; fidelity, certificates and PSNR use original floats. Periodic boundaries connect opposite edges and can cause wraparound artifacts.",
        "",
        "| Image | Penalty | Method | PSNR mean ± fit SD | Mean iterations | Mean seconds |",
        "|---|---:|---|---:|---:|---:|",
    ]
    for keys, values in frame.groupby(["dataset", "penalty", "method"]):
        case, lam, method = keys
        lines.append(
            f"| {case} | {lam:g} | {method} | {values.psnr.mean():.3f} ± {values.psnr.std():.3f} | {values.iterations.mean():.1f} | {values.elapsed_seconds.mean():.4f} |"
        )
    for case in config["datasets"]:
        fig, axes = plt.subplots(1, 2, figsize=(10, 4), layout="constrained")
        with np.load(
            f"assets/source_data/primal_dual/{case}_seed0_lambda0.06.npz", allow_pickle=False
        ) as data:
            for method in config["methods"]:
                h = data[method + "_history"]
                axes[0].semilogy(
                    h[:, 0], np.maximum(h[:, 3] / data["truth"].size, 1e-14), label=method
                )
                axes[1].semilogy(
                    h[:, 4], np.maximum(h[:, 3] / data["truth"].size, 1e-14), label=method
                )
        axes[0].set(xlabel="Iterations, unequal cost", ylabel="Feasible gap per pixel")
        axes[1].set(xlabel="Seconds including setup", ylabel="Feasible gap per pixel")
        axes[0].legend(fontsize=7)
        for suffix in ["png", "pdf"]:
            fig.savefig(f"assets/figures/primal_dual_{case}.{suffix}", dpi=170)
        plt.close(fig)
        lines.extend(["", f"![{case} convergence](../assets/figures/primal_dual_{case}.png)"])
    lines.extend(
        [
            "",
            "Every method also solves a separate 16×16 periodic crop, checked against an independent CLARABEL cone model. The crop oracle does not certify the larger image; the feasible dual gap does. Strong convexity makes the image optimum unique, but finite stopping tolerance permits small output differences. Accelerated methods need not win in a finite problem and tolerance regime.",
            "",
            "See [all fits](primal_dual_fits.csv), [the source](../src/optimization_lab/primal_dual.py), [validation](extensions_validation.json), and [the interactive explorer](explorer.html). Photo credit and dataset terms are in [third-party attribution](third_party.md).",
            "",
            "Primary algorithm reference: [Chambolle and Pock](https://www.cmap.polytechnique.fr/preprint/repository/685.pdf).",
        ]
    )
    Path("docs/primal_dual.md").write_text("\n".join(lines) + "\n")
    return dict(
        protocols=18,
        solver_results=72,
        all_certificates_feasible=True,
        independent_small_cone_oracles=True,
    )


def report_search():
    config = json.loads(Path("configs/adaptive_search.json").read_text())
    rows = []
    histories = {}
    for seed in config["seeds"]:
        for strategy in config["strategies"]:
            record, metrics = check_record(
                Path("results/reference/adaptive_search") / strategy / f"seed{seed}"
            )
            assert all(record["config"][key] == value for key, value in config.items())
            trials = metrics["trials"]
            assert len(trials) == config["trials"] and [t["number"] for t in trials] == list(
                range(config["trials"])
            )
            assert metrics["data_roles"] == dict(
                fit=64000, validation=16000, test=20000, users=943, items=1682
            )
            selected = min(trials, key=lambda t: (t["validation_rmse"], t["number"]))
            assert (
                selected["number"] == metrics["selected_trial"]
                and selected["parameters"] == metrics["selected_parameters"]
            )
            np.testing.assert_allclose(
                selected["validation_rmse"], metrics["validation_rmse"], atol=1e-12, rtol=0
            )
            root = Path("assets/source_data/adaptive_search") / strategy
            for trial in trials:
                path = root / f"seed{seed}_trial{trial['number']}.npz"
                assert sha256(path) == trial["curve_sha256"]
                with np.load(path, allow_pickle=False) as data:
                    h = data["history"]
                    assert h.shape == (config["epochs"] + 1, 3) and np.isfinite(h).all()
                    np.testing.assert_array_equal(h[:, 0], np.arange(config["epochs"] + 1))
                    assert int(np.argmin(h[:, 2])) == trial["selected_epoch"]
                    np.testing.assert_allclose(
                        h[:, 2].min(), trial["validation_rmse"], atol=1e-12, rtol=0
                    )
                assert trial["generator_updates"] == config["epochs"] * int(
                    np.ceil(64000 / config["batch_size"])
                )
            path = root / f"seed{seed}_test.npz"
            assert sha256(path) == metrics["test_evidence_sha256"]
            with np.load(path, allow_pickle=False) as data:
                assert data["user_rating_count"].sum() == 20000
                rmse = float(np.sqrt(data["user_squared_error"].sum() / 20000))
                mae = float(data["user_absolute_error"].sum() / 20000)
                np.testing.assert_allclose(
                    [rmse, mae],
                    [metrics["test"]["rmse"], metrics["test"]["mae"]],
                    atol=1e-12,
                    rtol=0,
                )
            histories[strategy, seed] = trials
            rows.append(
                dict(
                    strategy=strategy,
                    seed=seed,
                    selected_trial=metrics["selected_trial"],
                    selected_epoch=metrics["selected_epoch"],
                    validation_rmse=metrics["validation_rmse"],
                    test_rmse=rmse,
                    test_mae=mae,
                    **{f"selected_{key}": value for key, value in selected["parameters"].items()},
                )
            )
        assert [t["parameters"] for t in histories["random", seed][: config["startup_trials"]]] == [
            t["parameters"] for t in histories["tpe", seed][: config["startup_trials"]]
        ]
    assert len(rows) == 6
    frame = pd.DataFrame(rows)
    frame.to_csv("docs/adaptive_search_fits.csv", index=False)
    fig, axes = plt.subplots(1, 3, figsize=(12, 4), layout="constrained")
    for seed, ax in enumerate(axes):
        for strategy in config["strategies"]:
            curve = np.minimum.accumulate([t["validation_rmse"] for t in histories[strategy, seed]])
            ax.step(np.arange(1, len(curve) + 1), curve, where="post", label=strategy)
        ax.axvline(config["startup_trials"], ls="--", c="gray")
        ax.set(
            title=f"Fitting/search seed {seed}",
            xlabel="Completed candidate fits",
            ylabel="Best validation RMSE",
        )
    axes[0].legend()
    for suffix in ["png", "pdf"]:
        fig.savefig(f"assets/figures/adaptive_search.{suffix}", dpi=170)
    plt.close(fig)
    lines = [
        "# Adaptive model search on MovieLens",
        "",
        "Sequential Optuna TPE and random search optimize the same degree-regularized biased factor model using Adam. All 96 candidate fits complete 40 epochs; each strategy gets 16 candidates per seed. The first eight parameter proposals match exactly. Subsequent TPE proposals use validation outcomes; random proposals do not. Rank, factor/bias penalties and learning rate vary. Candidate budgets and model compute are explicit, not equal FLOPs.",
        "",
        "Fit/validation/test are 64,000/16,000/20,000 official MovieLens-100K ratings, preserving the earlier fixed split. The official test is scored only after a strategy selects its best validation candidate and epoch. This compares search procedures with a shared model family; it does not compare recommender architectures or measure chronological user behavior.",
        "",
        "| Search | Test RMSE mean ± fit SD | Validation RMSE |",
        "|---|---:|---:|",
    ]
    for strategy, values in frame.groupby("strategy"):
        lines.append(
            f"| {strategy} | {values.test_rmse.mean():.5f} ± {values.test_rmse.std():.5f} | {values.validation_rmse.mean():.5f} |"
        )
    lines.extend(
        [
            "",
            "![Search trajectories](../assets/figures/adaptive_search.png)",
            "",
            "Three-seed variation combines changes in factor initialization, batch order and proposal seed. Curves are validation measurements; the table contains the one selected model per campaign. Limited trials and split reuse can make selection optimistic. Neither a Bayesian-search label nor a modern library guarantees improvement. Partial searches reject continuation to avoid inventing sampler-RNG reproducibility; complete records skip unchanged. Use a fresh `--output` for a new campaign.",
            "",
            "```bash",
            "python scripts/run_adaptive_search.py --seed 0 --strategy tpe --device cuda:0",
            "python scripts/report_extensions.py",
            "```",
            "",
            "See [selected fits](adaptive_search_fits.csv), [validation](extensions_validation.json), [the original factor objective](../src/optimization_lab/recommendation.py), and [Optuna TPE documentation](https://optuna.readthedocs.io/en/stable/reference/samplers/generated/optuna.samplers.TPESampler.html). MovieLens ratings and factor checkpoints remain private.",
        ]
    )
    Path("docs/adaptive_search.md").write_text("\n".join(lines) + "\n")
    return dict(
        protocols=6,
        candidate_fits=96,
        shared_startup_proposals=True,
        selection_recomputed=True,
        test_metrics_recomputed=True,
    )


def main():
    tv = report_tv()
    adaptive = report_search()
    write_json(
        "docs/extensions_validation.json",
        dict(status="passed", primal_dual=tv, adaptive_search=adaptive),
    )
    print("Verified 72 TV results and all 96 adaptive-search candidates")


if __name__ == "__main__":
    main()
