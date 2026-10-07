"""Measured convergence and numerical certificates, with independent convex oracles."""

import argparse
import json
import time
from pathlib import Path

import cvxpy as cp
import numpy as np
from scipy.special import gammaln, xlogy
from sklearn.linear_model import Lasso

from .artifacts import Run
from .convex_cases import solve
from .first_order import lasso, nuclear_completion, poisson_smooth
from .resources import config_path


def poisson_score(rate, counts, exposure):
    predicted = exposure * rate
    strict = np.sum(predicted - xlogy(counts, predicted) + gammaln(counts + 1))
    bounded = np.maximum(predicted, 1e-6)
    return dict(
        strict_nll=float(strict) if np.isfinite(strict) else None,
        strict_nll_infinite=bool(not np.isfinite(strict)),
        declared_predictive_floor_nll=float(
            np.sum(bounded - xlogy(counts, bounded) + gammaln(counts + 1))
        ),
        predictive_count_floor=1e-6,
    )


def study(config, output="results/reference/first_order", quick=False):
    destination = Path(output) / "source_data" if quick else Path("assets/source_data/first_order")
    destination.mkdir(parents=True, exist_ok=True)
    seeds = config["seeds"][:1] if quick else config["seeds"]
    sources = [
        __file__,
        Path(__file__).with_name("first_order.py"),
        Path(__file__).with_name("convex_cases.py"),
    ]
    for seed in seeds:
        rng = np.random.default_rng(seed)
        run = Run(
            Path(output) / f"lasso_seed{seed}", config | dict(seed=seed, quick=quick), sources
        )
        cfg = config["lasso"]
        n, p = cfg["observations"], cfg["features"]
        covariance = cfg["correlation"] ** np.abs(np.arange(p)[:, None] - np.arange(p))
        a = rng.normal(size=(n, p)) @ np.linalg.cholesky(covariance).T / np.sqrt(n)
        truth = np.zeros(p)
        truth[rng.choice(p, 10, replace=False)] = rng.normal(0, 2, 10)
        b = a @ truth + rng.normal(0, 0.03, n)
        lam = cfg["lambda"]
        x = cp.Variable(p)
        problem = cp.Problem(cp.Minimize(0.5 * cp.sum_squares(a @ x - b) + lam * cp.norm1(x)))
        oracle = solve(problem)
        reference = x.value.copy()
        sklearn = Lasso(alpha=lam / n, fit_intercept=False, tol=1e-12, max_iter=100000).fit(a, b)
        arrays = dict(a=a, b=b, truth=truth, reference=reference, sklearn=sklearn.coef_)
        rows = []
        for method in ["ista", "fista", "admm"]:
            result = lasso(a, b, lam, method, cfg["steps"], cfg["tolerance"])
            arrays[f"{method}_x"] = result.pop("x")
            arrays[f"{method}_history"] = result.pop("history")
            result |= dict(
                method=method,
                objective_difference_to_oracle=result["certificate"]["objective"]
                - oracle["objective"],
            )
            rows.append(result)
        np.savez_compressed(destination / f"lasso_seed{seed}.npz", **arrays)
        run.finish(
            dict(
                rows=rows,
                oracle=oracle,
                sklearn_objective=float(
                    0.5 * np.sum((a @ sklearn.coef_ - b) ** 2) + lam * abs(sklearn.coef_).sum()
                ),
                scope="Owned ISTA/FISTA/ADMM use same convex objective, feasible dual bounds and stopping gap. sklearn alpha=lambda/n; no intercept. ADMM includes a Cholesky setup; iterations do not represent equal computational work.",
            )
        )
        print("Complete Lasso", seed, flush=True)

        run = Run(
            Path(output) / f"nuclear_seed{seed}", config | dict(seed=seed, quick=quick), sources
        )
        cfg = config["nuclear"]
        truth = (
            rng.normal(size=(cfg["rows"], cfg["rank"]))
            @ rng.normal(size=(cfg["rank"], cfg["columns"]))
            / np.sqrt(cfg["rank"])
        )
        y = truth + rng.normal(0, 0.05, truth.shape)
        mask = rng.random(truth.shape) < cfg["observed_fraction"]
        # Unobserved entries are not accessed by the fit or its certificate.
        observed = np.where(mask, y, np.nan)
        x = cp.Variable(truth.shape)
        objective = 0.5 * cp.sum_squares(cp.multiply(mask, x) - np.where(mask, y, 0)) + cfg[
            "lambda"
        ] * cp.normNuc(x)
        oracle = solve(cp.Problem(cp.Minimize(objective)))
        arrays = dict(truth=truth, observed=observed, mask=mask, reference=x.value.copy())
        rows = []
        for accelerated in [False, True]:
            method = "fista" if accelerated else "ista"
            result = nuclear_completion(
                observed, mask, cfg["lambda"], cfg["steps"], cfg["tolerance"], accelerated
            )
            prediction = result.pop("x")
            arrays[f"{method}_x"] = prediction
            arrays[f"{method}_history"] = result.pop("history")
            result |= dict(
                method=method,
                unobserved_rmse_to_synthetic_truth=float(
                    np.sqrt(np.mean((prediction[~mask] - truth[~mask]) ** 2))
                ),
                objective_difference_to_oracle=result["certificate"]["objective"]
                - oracle["objective"],
            )
            rows.append(result)
        np.savez_compressed(destination / f"nuclear_seed{seed}.npz", **arrays)
        run.finish(
            dict(
                rows=rows,
                oracle=oracle,
                scope="Independent synthetic rank3 completion task with a uniform nuclear norm; this is not the degree-regularized biased MovieLens objective. Exact SVD at every proximal step and certificate. Missing NaN values never treated as observed zeros. Penalty fixed before heldout scoring.",
            )
        )
        print("Complete nuclear completion", seed, flush=True)

    parameters = json.loads(config_path("additional_parameters.json").read_text())
    counts = np.array(parameters["poisson"]["N"]).ravel()
    validation = np.array(parameters["poisson_validation"]["N_test"])
    tasks = [("original", counts, validation, None, 1.0, None)]
    for seed in seeds:
        rng = np.random.default_rng(500 + seed)
        truth = (
            2
            + 1.3 * np.cos(np.arange(24) * 2 * np.pi / 24)
            + 0.3 * np.sin(np.arange(24) * 6 * np.pi / 24)
        )
        counts, validation, test = rng.poisson(10 * truth, size=(3, 24))
        tasks.append((f"synthetic_seed{seed}", counts, validation, test, 10.0, truth))
    for label, counts, validation, test, exposure, truth in tasks:
        run = Run(
            Path(output) / f"poisson_{label}",
            config
            | dict(
                task=label, counts=counts, validation=validation, exposure=exposure, quick=quick
            ),
            sources,
        )
        cfg = config["poisson"]
        d = np.eye(len(counts)) - np.roll(np.eye(len(counts)), 1, axis=0)
        arrays = dict(counts=counts, validation=validation)
        if test is not None:
            arrays |= dict(test=test, truth=truth)
        rows = []
        predictions = []
        for rho in cfg["rho"]:
            result = poisson_smooth(counts, rho, cfg["steps"], cfg["tolerance"], exposure)
            rate = result.pop("rate")
            arrays[f"rho{rho}_rate"] = rate
            arrays[f"rho{rho}_history"] = result.pop("history")
            x = cp.Variable(len(counts), nonneg=True)
            problem = cp.Problem(
                cp.Minimize(cp.sum(cp.kl_div(counts, exposure * x)) + rho * cp.sum_squares(d @ x))
            )
            oracle = solve(problem)
            constant = np.sum(xlogy(counts, counts) - counts - counts * np.log(exposure))
            oracle["owned_convention_objective"] = oracle["objective"] - float(constant)
            arrays[f"rho{rho}_reference_rate"] = x.value
            result |= dict(
                rho=rho,
                oracle=oracle,
                objective_difference_to_oracle=result["objective"]
                - oracle["owned_convention_objective"],
                validation=poisson_score(rate, validation, exposure),
            )
            rows.append(result)
            predictions.append(rate)
        # Declared predictive floor is used for validation selection; infinite strict scores retained.
        chosen = min(
            range(len(rows)), key=lambda i: rows[i]["validation"]["declared_predictive_floor_nll"]
        )
        selected = dict(
            rho=rows[chosen]["rho"],
            validation=rows[chosen]["validation"],
            independent_test=None
            if test is None
            else poisson_score(predictions[chosen], test, exposure),
        )
        np.savez_compressed(destination / f"poisson_{label}.npz", **arrays)
        run.finish(
            dict(
                rows=rows,
                selected=selected,
                scope="Nonnegative cyclic Poisson rates, exact likelihood proximal step and accelerated quadratic smoothing. KKT uses projected gradient mapping, including zero-count active boundaries. Original second day is validation only; synthetic task reserves an independent test day. Fixed rho grid selected by validation likelihood using an explicit1e-6 predictive count floor; strict infinite likelihood is retained separately.",
            )
        )
        print("Complete Poisson", label, flush=True)

    run = Run(Path(output) / "quadratic", config | dict(quick=quick), sources)
    cfg = config["quadratic"]
    arrays = {}
    rows = []
    rng = np.random.default_rng(23)
    q, _ = np.linalg.qr(rng.normal(size=(cfg["dimension"], cfg["dimension"])))
    optimum = rng.normal(size=cfg["dimension"])
    for condition in cfg["condition_numbers"]:
        values = np.geomspace(1, condition, cfg["dimension"])
        a = (q * values) @ q.T
        b = a @ optimum
        for method in ["gradient", "accelerated", "conjugate_gradient", "newton"]:
            x = np.zeros(len(b))
            z = x.copy()
            residual = b.copy()
            direction = residual.copy()
            history = []
            beginning = time.perf_counter()
            beta = (np.sqrt(condition) - 1) / (np.sqrt(condition) + 1)
            for iteration in range(cfg["steps"] + 1):
                error = x - optimum
                objective_gap = 0.5 * error @ a @ error
                history.append(
                    (
                        iteration,
                        objective_gap,
                        np.linalg.norm(a @ x - b),
                        time.perf_counter() - beginning,
                    )
                )
                if np.linalg.norm(a @ x - b) <= 1e-8 or iteration == cfg["steps"]:
                    break
                if method == "newton":
                    x += np.linalg.solve(a, b - a @ x)
                elif method == "conjugate_gradient":
                    ad = a @ direction
                    rr = residual @ residual
                    alpha = rr / (direction @ ad)
                    x += alpha * direction
                    new = residual - alpha * ad
                    direction = new + (new @ new) / rr * direction
                    residual = new
                else:
                    new = z - (a @ z - b) / condition
                    z = new + (beta if method == "accelerated" else 0) * (new - x)
                    x = new
            arrays[f"{condition}_{method}_history"] = np.array(history)
            rows.append(
                dict(
                    condition=condition,
                    method=method,
                    iterations=iteration,
                    objective_gap=float(objective_gap),
                    gradient_norm=float(history[-1][2]),
                    elapsed_seconds=history[-1][3],
                    reached_tolerance=bool(history[-1][2] <= 1e-8),
                )
            )
    np.savez_compressed(destination / "quadratic.npz", **arrays)
    run.finish(
        dict(
            rows=rows,
            scope="Strongly convex quadratics with known optimum. Gradient methods know spectral bounds; conjugate gradient accesses Hessian products and Newton solves with the full Hessian. These methods have different information and per-step costs. Slow or budget-limited convergence is retained.",
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/first_order.json")
    parser.add_argument("--output", default="results/reference/first_order")
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text())
    if args.quick:
        for cfg in config.values():
            if isinstance(cfg, dict) and "steps" in cfg:
                cfg["steps"] = 100
    study(config, args.output, args.quick)
