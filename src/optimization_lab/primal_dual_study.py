"""Real-photo and procedural TV comparisons with independently solved conic crops."""

import json
from pathlib import Path

import cvxpy as cp
import numpy as np
from scipy.ndimage import zoom

from .artifacts import Run, sha256
from .convex_cases import solve
from .primal_dual import denoise
from .resources import photo_path


def image(case, size):
    """Return one fixed grayscale truth; procedural data are explicitly identified."""
    if case == "photograph":
        rgb = np.load(photo_path(), allow_pickle=False)["image"]
        gray = rgb @ np.array([0.2126, 0.7152, 0.0722])
        return zoom(gray, (size / gray.shape[0], size / gray.shape[1]), order=1)
    y, x = np.mgrid[:size, :size] / size
    return (
        0.15
        + 0.5 * ((x - 0.3) ** 2 + (y - 0.3) ** 2 < 0.04)
        + 0.3 * ((x > 0.55) & (x < 0.85) & (y > 0.5) & (y < 0.8))
    )


def conic_reference(b, lam):
    """Independent isotropic TV cone model with the same periodic boundary."""
    x = cp.Variable(b.shape)
    vertical = cp.vstack([x[1:, :] - x[:-1, :], x[0:1, :] - x[-1:, :]])
    horizontal = cp.hstack([x[:, 1:] - x[:, :-1], x[:, 0:1] - x[:, -1:]])
    fields = cp.vstack(
        [
            cp.reshape(vertical, (1, b.size), order="C"),
            cp.reshape(horizontal, (1, b.size), order="C"),
        ]
    )
    program = cp.Problem(
        cp.Minimize(0.5 * cp.sum_squares(x - b) + lam * cp.sum(cp.norm(fields, 2, axis=0)))
    )
    result = solve(program)
    return result, x.value


def study(config, output="results/reference/primal_dual", quick=False):
    """Preserve every method, seed and declared penalty, including nonconvergence."""
    public = Path(output) / "source_data" if quick else Path("assets/source_data/primal_dual")
    public.mkdir(parents=True, exist_ok=True)
    root = Path(output)
    for case in config["datasets"]:
        truth = image(case, config["size"])
        for seed in config["seeds"]:
            observed = truth + np.random.default_rng(seed).normal(
                0, config["noise_standard_deviation"], truth.shape
            )
            for lam in config["lambdas"]:
                name = f"{case}_seed{seed}_lambda{lam:g}"
                directory = root / name
                protocol = config | dict(
                    dataset=case,
                    seed=seed,
                    penalty=lam,
                    quick=quick,
                    photo_sha256=sha256(photo_path()) if case == "photograph" else None,
                )
                if (directory / "run.json").exists():
                    prior = json.loads((directory / "run.json").read_text())
                    assert prior["config"] == protocol
                    if prior["status"] == "complete":
                        for source, digest in prior["source_sha256"].items():
                            assert sha256(Path(__file__).with_name(source)) == digest
                        print("Preserved", name, flush=True)
                        continue
                run = Run(
                    directory,
                    protocol,
                    [
                        __file__,
                        Path(__file__).with_name("primal_dual.py"),
                        Path(__file__).with_name("convex_cases.py"),
                    ],
                )
                small = observed[: config["oracle_size"], : config["oracle_size"]]
                oracle, oracle_image = conic_reference(small, lam)
                arrays = dict(
                    truth=truth, observed=observed, oracle_observed=small, oracle_image=oracle_image
                )
                rows = []
                for method in config["methods"]:
                    result = denoise(
                        observed,
                        lam,
                        method,
                        config["steps"],
                        config["gap_per_pixel"],
                        config["record_every"],
                    )
                    small_result = denoise(
                        small, lam, method, config["steps"], 1e-9, config["record_every"]
                    )
                    for key in ["x", "dual", "history", "snapshots", "snapshot_iterations"]:
                        arrays[f"{method}_{key}"] = result.pop(key)
                    for key in ["x", "dual"]:
                        arrays[f"{method}_oracle_{key}"] = small_result[key]
                    mse = float(np.mean((arrays[f"{method}_x"] - truth) ** 2))
                    result |= dict(
                        method=method,
                        psnr=float(-10 * np.log10(max(mse, 1e-16))),
                        mse=mse,
                        oracle_certificate=small_result["certificate"],
                        oracle_objective_difference=small_result["certificate"]["primal"]
                        - oracle["objective"],
                        oracle_image_max_absolute_difference=float(
                            np.max(abs(small_result["x"] - oracle_image))
                        ),
                    )
                    rows.append(result)
                    print(
                        name,
                        method,
                        result["iterations"],
                        result["certificate"]["gap"],
                        result["psnr"],
                        flush=True,
                    )
                path = public / (name + ".npz")
                np.savez_compressed(path, **arrays)
                run.finish(
                    dict(
                        rows=rows,
                        oracle=oracle,
                        evidence=str(path),
                        evidence_sha256=sha256(path),
                        observation_psnr=float(-10 * np.log10(np.mean((observed - truth) ** 2))),
                        scope=config["scope"],
                    )
                )
