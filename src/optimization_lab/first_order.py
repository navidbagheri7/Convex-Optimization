"""Owned proximal algorithms with genuinely dual-feasible objective certificates."""

import time

import numpy as np
from scipy.linalg import cho_factor, cho_solve


def soft_threshold(x, threshold):
    if threshold < 0 or not np.isfinite(threshold):
        raise ValueError("Nonnegative threshold required")
    return np.sign(x) * np.maximum(np.abs(x) - threshold, 0.0)


def lasso_certificate(a, b, x, lam):
    if lam <= 0:
        raise ValueError("Positive regularization required for this certificate")
    residual = a @ x - b
    primal = 0.5 * (residual @ residual) + lam * np.abs(x).sum()
    dual = residual / max(1.0, np.max(np.abs(a.T @ residual)) / lam) / (1 + 1e-12)
    objective = -0.5 * (dual @ dual) - b @ dual
    return {
        "objective": float(primal),
        "dual_objective": float(objective),
        "gap": float(primal - objective),
        "dual_feasibility": float(np.max(np.abs(a.T @ dual)) - lam),
    }


def lasso(a, b, lam, method="fista", steps=3000, tolerance=1e-7, rho=1.0):
    a, b = np.asarray(a, float), np.asarray(b, float)
    if (
        a.ndim != 2
        or b.shape != (a.shape[0],)
        or not np.isfinite(a).all()
        or not np.isfinite(b).all()
    ):
        raise ValueError("Finite design and target required")
    if (
        method not in ["ista", "fista", "admm"]
        or lam <= 0
        or steps < 1
        or rho <= 0
        or not np.isfinite([lam, rho, tolerance]).all()
    ):
        raise ValueError("Valid proximal method and positive parameters required")
    lipschitz = np.linalg.norm(a, 2) ** 2
    if lipschitz <= 0:
        raise ValueError("Nonzero design required")
    x = np.zeros(a.shape[1])
    z = x.copy()
    u = x.copy()
    momentum = 1.0
    history = []
    start = time.perf_counter()
    if method == "admm":
        factor = cho_factor(a.T @ a + rho * np.eye(a.shape[1]))
        atb = a.T @ b
    for step in range(steps + 1):
        deployed = z if method == "admm" else x
        certificate = lasso_certificate(a, b, deployed, lam)
        if step % 10 == 0 or certificate["gap"] <= tolerance or step == steps:
            history.append(
                (
                    step,
                    certificate["objective"],
                    certificate["dual_objective"],
                    certificate["gap"],
                    time.perf_counter() - start,
                )
            )
        if certificate["gap"] <= tolerance or step == steps:
            break
        if method == "admm":
            x = cho_solve(factor, atb + rho * (z - u))
            previous = z.copy()
            z = soft_threshold(x + u, lam / rho)
            u += x - z
            primal_residual = np.linalg.norm(x - z)
            dual_residual = rho * np.linalg.norm(z - previous)
        else:
            new = soft_threshold(z - a.T @ (a @ z - b) / lipschitz, lam / lipschitz)
            next_momentum = (1 + np.sqrt(1 + 4 * momentum**2)) / 2 if method == "fista" else 1.0
            z = new + (momentum - 1) / next_momentum * (new - x)
            x = new
            momentum = next_momentum
    return {
        "x": deployed.copy(),
        "history": np.array(history),
        "iterations": step,
        "certificate": certificate,
        "primal_residual": float(primal_residual) if method == "admm" and step else None,
        "dual_residual": float(dual_residual) if method == "admm" and step else None,
    }


def nuclear_certificate(x, y, mask, lam):
    residual = np.where(mask, x - y, 0.0)
    primal = 0.5 * np.sum(residual**2) + lam * np.linalg.svd(x, compute_uv=False).sum()
    spectral = np.linalg.norm(residual, 2)
    dual = residual / max(1.0, spectral / lam) / (1 + 1e-12)
    objective = -0.5 * np.sum(dual**2) - np.sum(dual * np.where(mask, y, 0.0))
    return {
        "objective": float(primal),
        "dual_objective": float(objective),
        "gap": float(primal - objective),
        "dual_spectral_norm": float(np.linalg.norm(dual, 2)),
    }


def nuclear_completion(y, mask, lam, steps=2000, tolerance=1e-6, accelerated=True):
    y = np.asarray(y, float)
    mask = np.asarray(mask, bool)
    if (
        y.ndim != 2
        or mask.shape != y.shape
        or not np.isfinite(y[mask]).all()
        or lam <= 0
        or not np.isfinite([lam, tolerance]).all()
        or steps < 1
    ):
        raise ValueError("Finite observed matrix, positive penalty and budget required")
    observed = np.where(mask, y, 0.0)
    x = np.zeros_like(observed)
    z = x.copy()
    momentum = 1.0
    history = []
    start = time.perf_counter()
    for step in range(steps + 1):
        cert = nuclear_certificate(x, observed, mask, lam)
        if step % 10 == 0 or cert["gap"] <= tolerance or step == steps:
            history.append(
                (
                    step,
                    cert["objective"],
                    cert["dual_objective"],
                    cert["gap"],
                    time.perf_counter() - start,
                )
            )
        if cert["gap"] <= tolerance or step == steps:
            break
        gradient = np.where(mask, z - observed, 0.0)
        u, s, vt = np.linalg.svd(z - gradient, full_matrices=False)
        new = (u * np.maximum(s - lam, 0.0)) @ vt
        next_momentum = (1 + np.sqrt(1 + 4 * momentum**2)) / 2 if accelerated else 1.0
        z = new + (momentum - 1) / next_momentum * (new - x)
        x = new
        momentum = next_momentum
    return {"x": x, "history": np.array(history), "certificate": cert, "iterations": step}


def poisson_prox(v, counts, step, exposure=1.0):
    v, counts = np.broadcast_arrays(np.asarray(v, float), np.asarray(counts, float))
    if (
        np.any(counts < 0)
        or step <= 0
        or exposure <= 0
        or not np.isfinite(v).all()
        or not np.isfinite(counts).all()
        or not np.isfinite([step, exposure]).all()
    ):
        raise ValueError("Finite values, nonnegative counts and positive parameters required")
    d = v - step * exposure
    root = np.hypot(d, 2 * np.sqrt(step * counts))
    result = np.asarray(0.5 * (d + root)).copy()
    negative = (d < 0) & (counts > 0)
    result[negative] = 2 * step * counts[negative] / (root[negative] - d[negative])
    return result


def poisson_smooth(counts, rho, steps=20000, tolerance=1e-7, exposure=1.0):
    from scipy.special import xlogy

    counts = np.asarray(counts, float).reshape(-1)
    if (
        rho < 0
        or exposure <= 0
        or np.any(counts < 0)
        or not np.isfinite(counts).all()
        or not len(counts)
        or not np.isfinite([rho, exposure, tolerance]).all()
    ):
        raise ValueError("Nonnegative counts and smoothing, positive exposure required")
    d = np.eye(len(counts)) - np.roll(np.eye(len(counts)), 1, axis=0)
    rate = np.full(len(counts), max(counts.mean() / exposure, 1e-3))
    z = rate.copy()
    momentum = 1.0
    history = []
    step_size = 1 / max(8 * rho, 1.0)
    for iteration in range(steps + 1):
        ratio = np.divide(counts, rate, out=np.zeros_like(rate), where=rate > 0)
        gradient = exposure - ratio + 2 * rho * d.T @ (d @ rate)
        residual = rate - np.maximum(0.0, rate - gradient)
        kkt = float(np.max(np.abs(residual)))
        objective = float(
            np.sum(exposure * rate - xlogy(counts, rate)) + rho * np.sum((d @ rate) ** 2)
        )
        if iteration % 50 == 0 or kkt <= tolerance or iteration == steps:
            history.append((iteration, objective, kkt))
        if kkt <= tolerance or iteration == steps:
            break
        new = poisson_prox(z - step_size * 2 * rho * d.T @ (d @ z), counts, step_size, exposure)
        next_momentum = (1 + np.sqrt(1 + 4 * momentum**2)) / 2
        z = new + (momentum - 1) / next_momentum * (new - rate)
        rate = new
        momentum = next_momentum
    return {
        "rate": rate,
        "objective": objective,
        "kkt_residual": kkt,
        "history": np.array(history),
        "iterations": iteration,
    }
