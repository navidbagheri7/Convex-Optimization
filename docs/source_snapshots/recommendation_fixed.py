"""Matched degree-regularized biased matrix factorization on official MovieLens splits.

All validation and deployment predictions use portable float64 NumPy arithmetic
on exported factors, including factors fitted in GPU float32. Raw ratings stay local.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn

from .artifacts import Run, sha256


def load_ratings(data_root, validation_fraction=0.2, split_seed=314159):
    root = Path(data_root) / "optimization/movielens/ml-100k"
    base = np.loadtxt(root / "u1.base", dtype=np.int64)
    test = np.loadtxt(root / "u1.test", dtype=np.int64)
    if len(base) != 80000 or len(test) != 20000:
        raise ValueError("Official u1.base/u1.test sizes required")
    for values in [base, test]:
        if np.any(values[:, :2] < 1) or np.any((values[:, 2] < 1) | (values[:, 2] > 5)):
            raise ValueError("Invalid official ratings")

    def keys(values):
        return set(map(tuple, values[:, :2]))

    if keys(base) & keys(test):
        raise ValueError("Base and test contain overlapping rated pairs")
    permutation = np.random.default_rng(split_seed).permutation(len(base))
    nval = int(len(base) * validation_fraction)
    fit = base[permutation[nval:]].copy()
    validation = base[permutation[:nval]].copy()
    for values in [fit, validation, test]:
        values[:, :2] -= 1
    users = int(max(base[:, 0].max(), test[:, 0].max()))
    items = int(max(base[:, 1].max(), test[:, 1].max()))
    return {
        "fit": fit,
        "validation": validation,
        "test": test,
        "users": users,
        "items": items,
        "mean": float(fit[:, 2].mean()),
        "user_degree": np.bincount(fit[:, 0], minlength=users),
        "item_degree": np.bincount(fit[:, 1], minlength=items),
        "source_sha256": {
            name: sha256(root / name) for name in ["u1.base", "u1.test", "u.data", "README"]
        },
    }


def initialize(data, rank, seed):
    rng = np.random.default_rng(seed)
    state = {
        "mean": data["mean"],
        "user": rng.normal(0, 0.05, (data["users"], rank)),
        "item": rng.normal(0, 0.05, (data["items"], rank)),
        "user_bias": np.zeros(data["users"]),
        "item_bias": np.zeros(data["items"]),
    }
    state["user"][data["user_degree"] == 0] = 0.0
    state["item"][data["item_degree"] == 0] = 0.0
    return state


def predict(state, users, items):
    users, items = np.asarray(users), np.asarray(items)
    if (
        users.shape != items.shape
        or not np.issubdtype(users.dtype, np.integer)
        or not np.issubdtype(items.dtype, np.integer)
    ):
        raise ValueError("Matching integer user/item IDs required")
    shape = users.shape
    users = users.ravel()
    items = items.ravel()
    result = np.full(len(users), float(state["mean"]))
    ku = (users >= 0) & (users < len(state["user"]))
    ki = (items >= 0) & (items < len(state["item"]))
    result[ku] += state["user_bias"][users[ku]]
    result[ki] += state["item_bias"][items[ki]]
    both = ku & ki
    result[both] += np.sum(state["user"][users[both]] * state["item"][items[both]], axis=1)
    return result.reshape(shape)


def objective(state, data, lam, bias_penalty):
    fit = data["fit"]
    residual = predict(state, fit[:, 0], fit[:, 1]) - fit[:, 2]
    factor = lam * (
        np.sum(data["user_degree"][:, None] * state["user"] ** 2)
        + np.sum(data["item_degree"][:, None] * state["item"] ** 2)
    )
    bias = bias_penalty * (
        np.sum(data["user_degree"] * state["user_bias"] ** 2)
        + np.sum(data["item_degree"] * state["item_bias"] ** 2)
    )
    return float((np.sum(residual**2) + factor + bias) / (2 * len(fit)))


def score(state, values):
    raw = predict(state, values[:, 0], values[:, 1])
    bounded = np.clip(raw, 1.0, 5.0)
    return {
        "rmse": float(np.sqrt(np.mean((bounded - values[:, 2]) ** 2))),
        "mae": float(np.mean(abs(bounded - values[:, 2]))),
        "unclipped_rmse": float(np.sqrt(np.mean((raw - values[:, 2]) ** 2))),
        "prediction": bounded,
    }


def fit_als(data, rank, seed, lam, bias_penalty, epochs):
    state = initialize(data, rank, seed)
    fit = data["fit"]
    history = []
    best = np.inf
    selected = None
    selected_epoch = 0
    user_rows = [np.flatnonzero(fit[:, 0] == u) for u in range(data["users"])]
    item_rows = [np.flatnonzero(fit[:, 1] == i) for i in range(data["items"])]
    for epoch in range(epochs + 1):
        train = objective(state, data, lam, bias_penalty)
        validation = score(state, data["validation"])["rmse"]
        history.append((epoch, train, validation))
        if validation < best:
            best = validation
            selected = {key: np.copy(value) for key, value in state.items()}
            selected_epoch = epoch
        if epoch == epochs:
            break
        for left, right, bias_left, bias_right, rows, column in [
            ("user", "item", "user_bias", "item_bias", user_rows, 1),
            ("item", "user", "item_bias", "user_bias", item_rows, 0),
        ]:
            for index, ids in enumerate(rows):
                if not len(ids):
                    continue
                neighbors = fit[ids, column]
                design = np.column_stack([np.ones(len(ids)), state[right][neighbors]])
                target = fit[ids, 2] - float(state["mean"]) - state[bias_right][neighbors]
                penalty = np.r_[bias_penalty, np.full(rank, lam)] * len(ids)
                coefficients = np.linalg.solve(
                    design.T @ design + np.diag(penalty), design.T @ target
                )
                state[bias_left][index] = coefficients[0]
                state[left][index] = coefficients[1:]
        if epoch % 10 == 0:
            print("ALS", seed, epoch, validation, flush=True)
    if np.any(np.diff(np.asarray(history)[:, 1]) > 1e-9):
        raise ValueError("ALS failed its matched-objective monotonicity check")
    return selected, np.asarray(history), selected_epoch


class FactorModel(nn.Module):
    def __init__(self, state):
        super().__init__()
        self.mean = float(state["mean"])
        for key in ["user", "item", "user_bias", "item_bias"]:
            setattr(self, key, nn.Parameter(torch.tensor(state[key], dtype=torch.float32)))

    def forward(self, u, i):
        return (
            self.mean + self.user_bias[u] + self.item_bias[i] + (self.user[u] * self.item[i]).sum(1)
        )

    def export(self):
        return {"mean": self.mean} | {
            key: getattr(self, key).detach().cpu().numpy().astype(np.float64)
            for key in ["user", "item", "user_bias", "item_bias"]
        }


def rating_loss(model, u, i, target, lam, bias_penalty):
    penalty = lam * (model.user[u].square().sum(1) + model.item[i].square().sum(1))
    penalty += bias_penalty * (model.user_bias[u].square() + model.item_bias[i].square())
    return 0.5 * ((model(u, i) - target).square() + penalty).mean()


def fit_stochastic(
    data, rank, seed, lam, bias_penalty, epochs, method, learning_rate, batch_size, device
):
    torch.manual_seed(seed)
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    rng = np.random.default_rng(seed)
    model = FactorModel(initialize(data, rank, seed)).to(device)
    optimizer = (
        torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
        if method == "sgd"
        else torch.optim.Adam(model.parameters(), lr=learning_rate)
    )
    fit = data["fit"]
    users = torch.tensor(fit[:, 0], device=device)
    items = torch.tensor(fit[:, 1], device=device)
    ratings = torch.tensor(fit[:, 2], dtype=torch.float32, device=device)
    history = []
    best = np.inf
    selected = None
    selected_epoch = 0
    for epoch in range(epochs + 1):
        state = model.export()
        train = objective(state, data, lam, bias_penalty)
        validation = score(state, data["validation"])["rmse"]
        if not np.isfinite([train, validation]).all():
            raise ValueError("Nonfinite factor model")
        history.append((epoch, train, validation))
        if validation < best:
            best = validation
            selected = {key: np.copy(value) for key, value in state.items()}
            selected_epoch = epoch
        if epoch == epochs:
            break
        order = rng.permutation(len(fit))
        for offset in range(0, len(fit), batch_size):
            ids = torch.as_tensor(order[offset : offset + batch_size], device=device)
            loss = rating_loss(model, users[ids], items[ids], ratings[ids], lam, bias_penalty)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        if epoch % 10 == 0:
            print(method, learning_rate, seed, epoch, validation, flush=True)
    return selected, np.asarray(history), selected_epoch


def test_statistics(state, data, seed):
    values = data["test"]
    metrics = score(state, values)
    errors = metrics.pop("prediction") - values[:, 2]
    count = np.bincount(values[:, 0], minlength=data["users"])
    sse = np.bincount(values[:, 0], weights=errors**2, minlength=data["users"])
    sae = np.bincount(values[:, 0], weights=abs(errors), minlength=data["users"])
    rng = np.random.default_rng(seed)
    draws = rng.integers(data["users"], size=(2000, data["users"]))
    bootstrap = np.sqrt(sse[draws].sum(1) / count[draws].sum(1))
    metrics["conditional_user_bootstrap_rmse_ci95"] = np.quantile(
        bootstrap, [0.025, 0.975]
    ).tolist()
    for label, mask in [
        ("warm", data["item_degree"][values[:, 1]] > 0),
        ("cold_item", data["item_degree"][values[:, 1]] == 0),
    ]:
        metrics[label] = {
            "n_ratings": int(mask.sum()),
            "rmse": float(np.sqrt(np.mean(errors[mask] ** 2))) if mask.any() else None,
        }
    return metrics, {
        "user_rating_count": count,
        "user_squared_error": sse,
        "user_absolute_error": sae,
    }


def fit(
    config, data_root, seed, device="cpu", quick=False, output="results/reference/recommendation"
):
    data = load_ratings(data_root, config["validation_fraction"], config["split_seed"])
    run = Run(
        Path(output) / f"seed{seed}",
        config
        | {
            "seed": seed,
            "device": device,
            "quick": quick,
            "source_data_sha256": data["source_sha256"],
        },
        [__file__],
    )
    destination = (
        run.directory / "source_data" if quick else Path("assets/source_data/recommendation")
    )
    destination.mkdir(parents=True, exist_ok=True)
    checkpoint = run.directory / "checkpoints"
    checkpoint.mkdir(exist_ok=True)
    rows = []
    for method in ["global_mean", "bias_only", "als", "sgd", "adam"]:
        candidates = []
        if method == "global_mean":
            state = initialize(data, 0, seed)
            candidates = [
                (
                    None,
                    state,
                    np.array(
                        [
                            [
                                0,
                                objective(state, data, config["lambda"], config["bias_penalty"]),
                                score(state, data["validation"])["rmse"],
                            ]
                        ]
                    ),
                    0,
                )
            ]
        elif method in ["bias_only", "als"]:
            rank = 0 if method == "bias_only" else config["rank"]
            state, history, epoch = fit_als(
                data, rank, seed, config["lambda"], config["bias_penalty"], config["epochs"]
            )
            candidates = [(None, state, history, epoch)]
        else:
            for lr in config["learning_rates"][method]:
                state, history, epoch = fit_stochastic(
                    data,
                    config["rank"],
                    seed,
                    config["lambda"],
                    config["bias_penalty"],
                    config["epochs"],
                    method,
                    lr,
                    config["batch_size"],
                    device,
                )
                candidates.append((lr, state, history, epoch))
        scores = [score(candidate[1], data["validation"])["rmse"] for candidate in candidates]
        chosen = int(np.argmin(scores))
        lr, state, history, epoch = candidates[chosen]
        for index, (candidate_lr, _, curve, _) in enumerate(candidates):
            np.savez_compressed(
                destination / f"{method}_{seed}_candidate{index}.npz", history=curve
            )
        path = checkpoint / f"{method}.npz"
        np.savez_compressed(
            path, **state, user_degree=data["user_degree"], item_degree=data["item_degree"]
        )
        metrics, statistics = test_statistics(state, data, seed)
        np.savez_compressed(destination / f"{method}_{seed}_test.npz", **statistics)
        rows.append(
            {
                "method": method,
                "seed": seed,
                "selected_learning_rate": lr,
                "selected_epoch": epoch,
                "validation_rmse": scores[chosen],
                "candidate_validation_rmse": scores,
                "test": metrics,
                "checkpoint_sha256": sha256(path),
                "selected_training_objective": objective(
                    state, data, config["lambda"], config["bias_penalty"]
                ),
            }
        )
    run.finish(
        {
            "rows": rows,
            "data_roles": {
                "fit": len(data["fit"]),
                "validation": len(data["validation"]),
                "test": len(data["test"]),
                "users": data["users"],
                "items": data["items"],
            },
            "protocol": "Official u1.base/u1.test, validation reserved inside base, fixed split across fitting seeds. Same degree-scaled regularized objective for ALS/SGD/Adam, same rank/lambda; two preset learning rates for stochastic methods, extra search work disclosed. Validation-only epoch/rate selection before each final test evaluation. NumPy float64 inference on exported factors. Ratings bounded to1..5 for validation/test, unclipped RMSE also reported. Public arrays aggregate errors per user, excluding raw ratings. Conditional bootstrap resamples users; it is distinct from three-fit training variability.",
        }
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--config", default="configs/recommendation.json")
    parser.add_argument("--output", default="results/reference/recommendation")
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text())
    if args.quick:
        config |= {"epochs": 2, "rank": 8, "learning_rates": {"sgd": [1.0], "adam": [0.003]}}
    fit(config, args.data_root, args.seed, args.device, args.quick, args.output)
