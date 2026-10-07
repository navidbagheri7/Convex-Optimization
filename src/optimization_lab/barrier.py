"""Feasible-start Newton barrier for reciprocal-fuel scheduling and dual witnesses."""

import numpy as np
from scipy.linalg import cho_factor, cho_solve
from scipy.optimize import linprog, nnls


def speed_constraints(parameters):
    d = np.asarray(parameters["d"], float)
    n = len(d)
    lower = d / np.asarray(parameters["smax"])
    upper = d / np.asarray(parameters["smin"])
    prefix = np.tril(np.ones((n, n)))
    g = np.vstack([-np.eye(n), np.eye(n), -prefix, prefix])
    h = np.concatenate(
        [-lower, upper, -np.asarray(parameters["tau_min"]), np.asarray(parameters["tau_max"])]
    )
    return d, parameters["a"] * d**2, float(parameters["c"]), g, h


def speed_certificate(t, k, c, g, h, constant=0.0, multipliers=None):
    if multipliers is None:
        raise ValueError("A supplied nonnegative dual witness is required")
    y = np.asarray(multipliers, float).copy()
    if np.any(y < 0) or not np.isfinite(y).all():
        raise ValueError("Finite nonnegative inequality multipliers required")
    coefficient = c + g.T @ y
    n = len(t)
    # Upper time-bound multipliers have identity columns. Repair tiny negative
    # conjugate coefficients without violating any dual sign constraint.
    y[n : 2 * n] += np.maximum(-coefficient, 0.0) + 1e-12
    coefficient = c + g.T @ y
    if np.min(coefficient) < 0:
        raise ValueError("Dual conjugate domain violated")
    primal = float(np.sum(k / t + c * t) + constant)
    dual = float(constant - h @ y + np.sum(2 * np.sqrt(k * coefficient)))
    return {
        "objective": primal,
        "dual_objective": dual,
        "gap": primal - dual,
        "constraint_violation": float(max(0, np.max(g @ t - h))),
        "stationarity_residual": float(np.max(abs(-k / t**2 + c + g.T @ y))),
        "complementarity_max": float(np.max(abs(y * (h - g @ t)))),
        "multipliers": y,
    }


def speed_barrier(parameters, final_mu=1e9):
    d, k, c, g, h = speed_constraints(parameters)
    n = len(d)
    phase = linprog(
        np.r_[np.zeros(n), -1.0],
        A_ub=np.column_stack([g, np.ones(len(h))]),
        b_ub=h,
        bounds=[(None, None)] * n + [(0, None)],
        method="highs",
    )
    if not phase.success or phase.x[-1] <= 1e-10:
        raise ValueError(
            "No certified strict interior; this feasible-start barrier is inapplicable"
        )
    t = phase.x[:n]
    initial_slack = float((h - g @ t).min())
    mu = 1.0
    history = []
    iterations = 0

    def objective(point, weight):
        slack = h - g @ point
        if np.any(slack <= 0) or np.any(point <= 0):
            return np.inf
        return float(np.sum(k / point + c * point) - np.log(slack).sum() / weight)

    while True:
        for _ in range(100):
            slack = h - g @ t
            y = 1 / (mu * slack)
            gradient = -k / t**2 + c + g.T @ y
            hessian = np.diag(2 * k / t**3) + g.T @ ((1 / (mu * slack**2))[:, None] * g)
            direction = cho_solve(cho_factor(hessian), -gradient)
            decrement = float(-gradient @ direction / 2)
            if decrement < 1e-11:
                break
            change = g @ direction
            positive = change > 0
            alpha = (
                min(1.0, 0.99 * np.min(slack[positive] / change[positive]))
                if positive.any()
                else 1.0
            )
            baseline = objective(t, mu)
            for _ in range(80):
                if objective(t + alpha * direction, mu) <= baseline + 0.01 * alpha * (
                    gradient @ direction
                ):
                    break
                alpha *= 0.5
            else:
                raise RuntimeError("Barrier line search failed")
            t = t + alpha * direction
            iterations += 1
        witness = speed_certificate(
            t, k, c, g, h, parameters["b"] * d.sum(), 1 / (mu * (h - g @ t))
        )
        history.append(
            (
                iterations,
                mu,
                witness["objective"],
                witness["dual_objective"],
                witness["gap"],
                witness["constraint_violation"],
                decrement,
            )
        )
        if mu >= final_mu:
            break
        mu = min(10 * mu, final_mu)
    central_witness = witness
    active = (h - g @ t) < 1e-5
    recovered = np.zeros(len(h))
    if active.any():
        recovered[active], _ = nnls(g[active].T, k / t**2 - c, maxiter=10000)
        refined = speed_certificate(t, k, c, g, h, parameters["b"] * d.sum(), recovered)
        if refined["gap"] < witness["gap"]:
            witness = refined
    return {
        "time": t,
        "speed": d / t,
        "certificate": witness,
        "history": np.array(history),
        "initial_strict_slack": initial_slack,
        "central_path_certificate": central_witness,
        "dual_recovery": "Nonnegative least-squares KKT fit on nearly active inequalities; independently checked feasible dual, retained only when its bound improves.",
    }
