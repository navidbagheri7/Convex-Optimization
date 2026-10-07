"""Spectral accuracy, multiplicity and approximation evidence."""

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA

from .artifacts import Run, sha256
from .resources import photo_path
from .spectral import channel_svd, nonsymmetric_schur, pca, symmetric_qr


def study(config, output="results/reference/spectral", quick=False):
    run = Run(
        output, config | dict(quick=quick), [__file__, Path(__file__).with_name("spectral.py")]
    )
    destination = Path(output) / "source_data" if quick else Path("assets/source_data/spectral")
    destination.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(config["seed"])
    n = config["matrix_dimension"]
    q, _ = np.linalg.qr(rng.normal(size=(n, n)))
    spectra = dict(
        repeated=np.r_[np.full(4, -2.0), np.zeros(4), np.arange(1, n - 7)],
        indefinite=np.linspace(-10, 10, n),
        ill_conditioned=np.geomspace(1e-8, 1, n),
    )
    rows = []
    arrays = {}
    for label, values in spectra.items():
        matrix = (q * values) @ q.T
        result = symmetric_qr(matrix, config["tolerance"], config["max_steps"])
        for key in ["values", "vectors", "history"]:
            arrays[f"{label}_{key}"] = result.pop(key)
        arrays[f"{label}_matrix"] = matrix
        result |= dict(
            case=label,
            maximum_eigenvalue_error=float(
                np.max(abs(arrays[f"{label}_values"] - np.linalg.eigvalsh(matrix)))
            ),
        )
        rows.append(result)
    matrix = np.array([[1.0, 100.0, 0.0], [0.0, 2.0, 10.0], [0.0, 0.0, 3.0]])
    schur = nonsymmetric_schur(matrix)
    schur["incorrect_individual_eigenvector_residual"] = float(
        np.linalg.norm(matrix @ schur["vectors"] - schur["vectors"] * schur["values"])
    )
    for key in ["triangular", "vectors", "values"]:
        arrays[f"schur_{key}"] = schur.pop(key)
    np.savez_compressed(destination / "qr.npz", **arrays)

    source_photo = photo_path()
    image = np.load(source_photo)["image"]
    compression, reconstructions = channel_svd(image, config["svd_ranks"])
    np.savez_compressed(destination / "svd.npz", original=image, **reconstructions)
    iris = load_iris()
    arrays = dict(features=iris.data, species=iris.target)
    geometry = []
    for standardized in [False, True]:
        result = pca(iris.data, config["pca_components"], standardized)
        normalized = (iris.data - result["mean"]) / result["scale"]
        reference = PCA(n_components=config["pca_components"], svd_solver="full").fit(normalized)
        projector_difference = np.linalg.norm(
            result["components"].T @ result["components"]
            - reference.components_.T @ reference.components_
        )
        label = "standardized" if standardized else "raw_units"
        arrays[f"{label}_scores"] = result["scores"]
        arrays[f"{label}_components"] = result["components"]
        arrays[f"{label}_reconstruction"] = result["reconstruction"]
        geometry.append(
            dict(
                standardized=standardized,
                explained_variance_ratio=result["explained_variance_ratio"],
                sklearn_projector_difference=float(projector_difference),
                raw_units_reconstruction_rmse=float(
                    np.sqrt(np.mean((result["reconstruction"] - iris.data) ** 2))
                ),
            )
        )
    np.savez_compressed(destination / "pca.npz", **arrays)
    run.finish(
        dict(
            qr=rows,
            schur=schur,
            compression=compression,
            pca=geometry,
            photo_sha256=sha256(source_photo),
            scope="Owned Householder reduction and shifted/deflated symmetric QR retain repeated eigenvalues. Nonsymmetric example uses library complex Schur and explicitly shows that Schur columns are not general eigenvectors. Channel SVD verifies the exact raw Frobenius tail identity; display is clipped to[0,1], with PSNR capped at160dB by the declared1e-16 MSE floor. SVD parameter counts are not a compressed-file codec. Iris PCA is descriptive geometry with raw and standardized units; sklearn subspace comparison avoids arbitrary sign choices.",
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/spectral.json")
    parser.add_argument("--output", default="results/reference/spectral")
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    study(json.loads(Path(args.config).read_text()), args.output, args.quick)
