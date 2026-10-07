"""Owned symmetric Householder/shifted QR and range-correct SVD/PCA."""

import numpy as np
from scipy.linalg import schur


def tridiagonalize(matrix):
    a = np.array(matrix, float, copy=True)
    if (
        a.ndim != 2
        or a.shape[0] != a.shape[1]
        or not len(a)
        or not np.isfinite(a).all()
        or not np.allclose(a, a.T, rtol=1e-12, atol=1e-12)
    ):
        raise ValueError("Finite nonempty symmetric matrix required")
    n = len(a)
    q = np.eye(n)
    for k in range(n - 2):
        x = a[k + 1 :, k].copy()
        length = np.linalg.norm(x)
        if length == 0:
            continue
        x[0] += np.copysign(length, x[0])
        x /= np.linalg.norm(x)
        h = np.eye(n)
        h[k + 1 :, k + 1 :] -= 2 * np.outer(x, x)
        a = h @ a @ h
        q = q @ h
    a = 0.5 * (a + a.T)
    a[np.abs(np.arange(n)[:, None] - np.arange(n)) > 1] = 0.0
    return a, q


def symmetric_qr(matrix, tolerance=1e-12, max_steps=10000):
    original = np.asarray(matrix, float)
    t, vectors = tridiagonalize(original)
    n = len(t)
    hi = n - 1
    history = []
    iterations = 0
    while hi > 0 and iterations < max_steps:
        for j in range(1, hi + 1):
            if abs(t[j, j - 1]) <= tolerance * (abs(t[j, j]) + abs(t[j - 1, j - 1]) + 1):
                t[j, j - 1] = t[j - 1, j] = 0.0
        if t[hi, hi - 1] == 0:
            hi -= 1
            continue
        lo = hi - 1
        while lo > 0 and t[lo, lo - 1] != 0:
            lo -= 1
        delta = (t[hi - 1, hi - 1] - t[hi, hi]) / 2
        off = t[hi, hi - 1]
        shift = t[hi, hi] - np.copysign(1.0, delta) * off**2 / (abs(delta) + np.hypot(delta, off))
        block = t[lo : hi + 1, lo : hi + 1]
        q, r = np.linalg.qr(block - shift * np.eye(len(block)))
        updated = r @ q + shift * np.eye(len(block))
        updated = 0.5 * (updated + updated.T)
        updated[np.abs(np.arange(len(block))[:, None] - np.arange(len(block))) > 1] = 0.0
        t[lo : hi + 1, lo : hi + 1] = updated
        vectors[:, lo : hi + 1] = vectors[:, lo : hi + 1] @ q
        iterations += 1
        history.append((iterations, np.max(abs(np.diag(t, -1)))))
    order = np.argsort(np.diag(t), kind="stable")
    values = np.diag(t)[order]
    vectors = vectors[:, order]
    return {
        "values": values,
        "vectors": vectors,
        "iterations": iterations,
        "converged": hi == 0,
        "history": np.array(history).reshape(-1, 2),
        "relative_eigen_residual": float(
            np.linalg.norm(original @ vectors - vectors * values)
            / max(1.0, np.linalg.norm(original))
        ),
        "orthogonality_residual": float(np.linalg.norm(vectors.T @ vectors - np.eye(n))),
    }


def nonsymmetric_schur(matrix):
    a = np.asarray(matrix, float)
    t, z = schur(a, output="complex")
    return {
        "triangular": t,
        "vectors": z,
        "values": np.diag(t),
        "relative_reconstruction_residual": float(
            np.linalg.norm(a - z @ t @ z.conj().T) / max(1.0, np.linalg.norm(a))
        ),
        "scope": "Schur vectors form an orthonormal invariant basis. They are not generally individual eigenvectors for nonnormal matrices.",
    }


def pca(x, components=2, standardize=False):
    x = np.asarray(x, float)
    if x.ndim != 2 or len(x) < 2 or not np.isfinite(x).all() or not 1 <= components <= min(x.shape):
        raise ValueError("Finite data and valid PCA dimension required")
    mean = x.mean(0)
    scale = x.std(0, ddof=1) if standardize else np.ones(x.shape[1])
    scale = np.where(scale > 0, scale, 1.0)
    centered = (x - mean) / scale
    u, s, vt = np.linalg.svd(centered, full_matrices=False)
    directions = vt[:components]
    scores = centered @ directions.T
    variance = s**2 / (len(x) - 1)
    return {
        "scores": scores,
        "components": directions,
        "mean": mean,
        "scale": scale,
        "explained_variance": variance[:components],
        "explained_variance_ratio": variance[:components] / variance.sum(),
        "reconstruction": scores @ directions * scale + mean,
    }


def channel_svd(image, ranks):
    image = np.asarray(image, float)
    if (
        image.ndim != 3
        or image.shape[2] != 3
        or np.any((image < 0) | (image > 1))
        or not np.isfinite(image).all()
    ):
        raise ValueError("RGB image in [0,1] required")
    decomposition = [np.linalg.svd(image[:, :, c], full_matrices=False) for c in range(3)]
    reconstructions = {}
    rows = []
    for rank in ranks:
        if not 1 <= rank <= min(image.shape[:2]):
            raise ValueError("Invalid SVD rank")
        raw = np.stack(
            [(u[:, :rank] * s[:rank]) @ vt[:rank] for u, s, vt in decomposition], axis=-1
        )
        bounded = np.clip(raw, 0.0, 1.0)
        squared = np.mean((bounded - image) ** 2)
        reconstructions[f"rank{rank}"] = bounded
        rows.append(
            {
                "rank": rank,
                "raw_squared_error": float(np.sum((raw - image) ** 2)),
                "eckart_young_tail_sum": float(
                    sum(np.sum(s[rank:] ** 2) for _, s, _ in decomposition)
                ),
                "display_clipped_mse": float(squared),
                "display_psnr": float(-10 * np.log10(max(squared, 1e-16))),
                "parameter_fraction": float(
                    3 * rank * (image.shape[0] + image.shape[1] + 1) / image.size
                ),
            }
        )
    return rows, reconstructions
