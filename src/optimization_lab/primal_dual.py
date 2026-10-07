"""Periodic isotropic TV: owned primal-dual, dual acceleration and FFT splitting.

The feasible dual certificate applies to every iterate, independently of method.
All scores use the unconstrained floating image; clipping is display-only.
"""

import time

import numpy as np


def gradient(x):
    """Periodic forward differences, with rows/columns as the two components."""
    x = np.asarray(x, dtype=float)
    return np.stack([np.roll(x, -1, axis=axis) - x for axis in [0, 1]])


def adjoint(p):
    """Exact Euclidean adjoint of the periodic forward-difference operator."""
    return sum(np.roll(p[axis], 1, axis=axis) - p[axis] for axis in [0, 1])


def project_ball(p, radius):
    """Project each two-component dual pixel onto an L2 ball."""
    return p / np.maximum(1.0, np.linalg.norm(p, axis=0) / radius)


def certificate(x, observed, p, lam):
    """Primal objective, feasible Fenchel dual lower bound and absolute gap."""
    p = project_ball(np.asarray(p, float), lam)
    y = adjoint(p)
    primal = 0.5 * np.sum((x - observed) ** 2) + lam * np.linalg.norm(gradient(x), axis=0).sum()
    dual = np.sum(y * observed) - 0.5 * np.sum(y**2)
    return dict(
        primal=float(primal),
        dual=float(dual),
        gap=float(primal - dual),
        dual_feasibility=float(max(0, np.linalg.norm(p, axis=0).max() - lam)),
    )


def denoise(observed, lam, method="pdhg", steps=8000, gap_per_pixel=1e-6, record_every=20, rho=1.0):
    """Solve one fixed convex ROF problem; stop on a feasible dual gap per pixel.

    Accelerated PDHG exploits the known unit strong convexity of the quadratic
    fidelity; its varying steps retain tau*sigma*||D||^2 < 1. FISTA operates on
    the smooth constrained dual. ADMM solves its quadratic step with the exact
    periodic FFT diagonalization. Timings include method setup.
    """
    b = np.asarray(observed, float)
    if b.ndim != 2 or min(b.shape) < 2 or not np.isfinite(b).all():
        raise ValueError("A finite two-dimensional image with each side at least two is required")
    if not np.isfinite([lam, gap_per_pixel, rho]).all() or min(lam, gap_per_pixel, rho) <= 0:
        raise ValueError("Finite positive penalty, gap tolerance and ADMM rho required")
    if (
        method not in ["pdhg", "accelerated_pdhg", "dual_fista", "admm"]
        or steps < 1
        or record_every < 1
    ):
        raise ValueError("Known method and positive iteration/recording budgets required")
    started = time.perf_counter()
    x, extrapolated = b.copy(), b.copy()
    p = np.zeros((2, *b.shape))
    tau = sigma = 0.99 / np.sqrt(8.0)
    z, scaled_dual = gradient(x), np.zeros_like(p)
    dual_extrapolated, momentum = p.copy(), 1.0
    eigen = None
    if method == "admm":
        eigen = (
            4 * np.sin(np.pi * np.arange(b.shape[0]) / b.shape[0])[:, None] ** 2
            + 4 * np.sin(np.pi * np.arange(b.shape[1]) / b.shape[1])[None, :] ** 2
        )
    history, snapshots, snapshot_iterations = [], [], []
    for iteration in range(steps + 1):
        if iteration in [0, 1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 4000, 8000]:
            snapshots.append(x.copy())
            snapshot_iterations.append(iteration)
        if iteration % record_every == 0 or iteration == steps:
            cert = certificate(x, b, p, lam)
            history.append(
                [
                    iteration,
                    cert["primal"],
                    cert["dual"],
                    cert["gap"],
                    time.perf_counter() - started,
                ]
            )
            if cert["gap"] <= gap_per_pixel * b.size or iteration == steps:
                if snapshot_iterations[-1] != iteration:
                    snapshots.append(x.copy())
                    snapshot_iterations.append(iteration)
                break
        if method in ["pdhg", "accelerated_pdhg"]:
            p = project_ball(p + sigma * gradient(extrapolated), lam)
            old = x
            x = (x - tau * adjoint(p) + tau * b) / (1 + tau)
            theta = 1 / np.sqrt(1 + 2 * tau) if method == "accelerated_pdhg" else 1.0
            extrapolated = x + theta * (x - old)
            tau, sigma = tau * theta, sigma / theta
        elif method == "dual_fista":
            old = p
            p = project_ball(
                dual_extrapolated + gradient(b - adjoint(dual_extrapolated)) / 8.0, lam
            )
            x = b - adjoint(p)
            next_momentum = (1 + np.sqrt(1 + 4 * momentum**2)) / 2
            dual_extrapolated = p + ((momentum - 1) / next_momentum) * (p - old)
            momentum = next_momentum
        else:
            rhs = b + rho * adjoint(z - scaled_dual)
            x = np.fft.ifft2(np.fft.fft2(rhs) / (1 + rho * eigen)).real
            updated = gradient(x) + scaled_dual
            norms = np.linalg.norm(updated, axis=0)
            z = updated * np.maximum(0, 1 - lam / (rho * np.maximum(norms, np.finfo(float).tiny)))
            scaled_dual += gradient(x) - z
            p = project_ball(rho * scaled_dual, lam)
    return dict(
        x=x,
        dual=p,
        history=np.array(history),
        snapshots=np.array(snapshots),
        snapshot_iterations=np.array(snapshot_iterations),
        certificate=certificate(x, b, p, lam),
        iterations=iteration,
        converged=bool(cert["gap"] <= gap_per_pixel * b.size),
        elapsed_seconds=time.perf_counter() - started,
        operator_squared_norm_bound=8.0,
    )
