"""Portable public-evidence audit, independent of private ratings/checkpoints."""

from pathlib import Path

import nbformat
import numpy as np
from scipy.special import xlogy

from .artifacts import sha256, write_json
from .first_order import lasso_certificate, nuclear_certificate
from .reporting import EXPECTED, read, records
from .routing import integer_one_tree, tour_length


def validate(require_notebooks=True):
    protocols, counts = records()
    for path in Path("configs").glob("*.json"):
        assert (
            path.read_bytes()
            == (Path("src/optimization_lab/default_configs") / path.name).read_bytes()
        ), f"Bundled configuration mismatch: {path}"
    for family, path, record, metrics in protocols:
        for name, digest in record["source_sha256"].items():
            source = Path("src/optimization_lab") / name
            if family == "recommendation_fixed" and name == "recommendation.py":
                source = Path("docs/source_snapshots/recommendation_fixed.py")
            assert sha256(source) == digest, f"Source changed: {path}, {source}"
    numeric = list(Path("assets/source_data").rglob("*.npz"))
    for path in numeric:
        with np.load(path, allow_pickle=False) as data:
            for key in data.files:
                values = data[key]
                if key == "observed" and path.stem.startswith("nuclear_"):
                    assert np.isfinite(values[data["mask"]]).all()
                    assert np.isnan(values[~data["mask"]]).all()
                else:
                    assert np.isfinite(values).all(), f"Nonfinite public array {path}: {key}"
    for seed in [0, 1, 2]:
        for family in ["lasso", "nuclear"]:
            root = Path("results/reference/first_order") / f"{family}_seed{seed}"
            record = read(root / "run.json")
            metrics = read(root / "metrics.json")
            data = np.load(Path("assets/source_data/first_order") / f"{family}_seed{seed}.npz")
            for row in metrics["rows"]:
                lam = record["config"][family]["lambda"]
                method = row["method"]
                cert = (
                    lasso_certificate(data["a"], data["b"], data[f"{method}_x"], lam)
                    if family == "lasso"
                    else nuclear_certificate(
                        data[f"{method}_x"], data["observed"], data["mask"], lam
                    )
                )
                for key in cert:
                    np.testing.assert_allclose(
                        cert[key], row["certificate"][key], atol=1e-9, rtol=0
                    )
                tolerance = 1e-7 if family == "lasso" else 1e-6
                assert -1e-10 <= cert["gap"] <= tolerance * 1.01
                if family == "lasso":
                    assert cert["dual_feasibility"] <= 1e-10
                else:
                    assert cert["dual_spectral_norm"] <= lam * (1 + 1e-10)
    for label in ["original", "synthetic_seed0", "synthetic_seed1", "synthetic_seed2"]:
        root = Path("results/reference/first_order") / f"poisson_{label}"
        record = read(root / "run.json")
        metrics = read(root / "metrics.json")
        exposure = record["config"]["exposure"]
        data = np.load(Path("assets/source_data/first_order") / f"poisson_{label}.npz")
        counts_ = data["counts"]
        d = np.eye(len(counts_)) - np.roll(np.eye(len(counts_)), 1, axis=0)
        for row in metrics["rows"]:
            rho = row["rho"]
            rate = data[f"rho{rho}_rate"]
            assert np.all(rate >= 0)
            objective = float(
                np.sum(exposure * rate - xlogy(counts_, rate)) + rho * np.sum((d @ rate) ** 2)
            )
            np.testing.assert_allclose(objective, row["objective"], atol=1e-9, rtol=0)
            ratio = np.divide(counts_, rate, out=np.zeros_like(rate), where=rate > 0)
            gradient = exposure - ratio + 2 * rho * d.T @ (d @ rate)
            kkt = np.max(abs(rate - np.maximum(0, rate - gradient)))
            np.testing.assert_allclose(kkt, row["kkt_residual"], atol=1e-9, rtol=0)
            assert kkt < 1.01e-7
            assert abs(row["objective_difference_to_oracle"]) < 1e-5
        chosen = min(
            metrics["rows"], key=lambda r: r["validation"]["declared_predictive_floor_nll"]
        )
        assert chosen["rho"] == metrics["selected"]["rho"]
        if label == "original":
            assert metrics["selected"]["independent_test"] is None
    recommendation_candidates = 0
    selected_models = 0
    for family in ["recommendation_fixed", "recommendation"]:
        for seed in [0, 1, 2]:
            root = Path("results/reference") / family / f"seed{seed}"
            metrics = read(root / "metrics.json")
            record = read(root / "run.json")
            assert record["config"]["seed"] == seed
            assert metrics["data_roles"] == dict(
                fit=64000, validation=16000, test=20000, users=943, items=1682
            )
            for row in metrics["rows"]:
                selected_models += 1
                method = row["method"]
                scores = row["candidate_validation_rmse"]
                index = int(np.argmin(scores))
                np.testing.assert_allclose(
                    row["validation_rmse"], scores[index], atol=1e-12, rtol=0
                )
                recommendation_candidates += len(scores)
                for candidate in range(len(scores)):
                    path = (
                        Path("assets/source_data")
                        / family
                        / f"{method}_{seed}_candidate{candidate}.npz"
                    )
                    data = np.load(path)
                    curve = data["history"]
                    assert len(curve) == (1 if method == "global_mean" else 81)
                    np.testing.assert_allclose(
                        curve[:, 2].min(), scores[candidate], atol=1e-12, rtol=0
                    )
                    if method in ["bias_only", "als"]:
                        assert np.all(np.diff(curve[:, 1]) <= 1e-9)
                selected = np.load(
                    Path("assets/source_data") / family / f"{method}_{seed}_candidate{index}.npz"
                )["history"]
                assert row["selected_epoch"] == int(np.argmin(selected[:, 2]))
                data = np.load(Path("assets/source_data") / family / f"{method}_{seed}_test.npz")
                assert data["user_rating_count"].sum() == 20000
                rmse = np.sqrt(data["user_squared_error"].sum() / 20000)
                mae = data["user_absolute_error"].sum() / 20000
                np.testing.assert_allclose(
                    [rmse, mae], [row["test"]["rmse"], row["test"]["mae"]], atol=1e-12, rtol=0
                )
    assert recommendation_candidates == 81 and selected_models == 30
    routes = 0
    for name in ["berlin52", "ch150"]:
        main = np.load(Path("assets/source_data/routing") / f"{name}.npz")
        distance = main["distance"]
        metrics = read(Path("results/reference/routing") / name / "metrics.json")
        witness = integer_one_tree(distance, main["one_tree_potentials"])
        assert witness["bound"] == metrics["one_tree_certificate"]["bound"]
        assert witness["bound"] <= metrics["published_optimum"]
        np.testing.assert_array_equal(witness["edges"], main["one_tree_edges"])
        for family in ["routing", "memetic"]:
            data = np.load(Path("assets/source_data") / family / f"{name}.npz")
            root = Path("results/reference") / family / name
            record = read(root / "run.json")
            m = read(root / "metrics.json")
            assert sorted(set(r["seed"] for r in m["rows"])) == [0, 1, 2]
            for row in m["rows"]:
                routes += 1
                method = row["method"]
                seed = row["seed"]
                value = tour_length(distance, data[f"{method}_seed{seed}_tour"])
                assert value == row["length"] and value >= metrics["published_optimum"]
                expected = 100 * (value / metrics["published_optimum"] - 1)
                np.testing.assert_allclose(
                    expected, row["gap_to_published_optimum_percent"], atol=1e-12, rtol=0
                )
                if method != "ortools_gls":
                    assert row["edge_work"] <= record["config"]["edge_budget"]
                    curve = data[f"{method}_seed{seed}_history"]
                    assert curve[-1, 1] == value and np.all(np.diff(curve[:, 1]) < 0)
                    assert curve[-1, 0] <= row["edge_work"]
    assert routes == 84
    exact = np.load("assets/source_data/exact/tsp12.npz")
    m = read("results/reference/exact/metrics.json")
    optimum = m["dynamic_programming"]["length"]
    assert optimum == 2248 and m["milp"]["proven_optimal"]
    assert (
        tour_length(exact["distance"], exact["dp_tour"])
        == tour_length(exact["distance"], exact["milp_tour"])
        == optimum
    )
    assert (
        integer_one_tree(exact["distance"], exact["bound_potentials"])["bound"]
        == m["integer_one_tree_bound"]
        == optimum
    )
    m = read("results/reference/convex/speed/metrics.json")
    assert (
        -1e-8 <= m["owned_certificate"]["gap"] < 1e-6
        and m["owned_certificate"]["constraint_violation"] < 1e-8
    )
    for row in read("results/reference/convex/regulator/metrics.json")["rows"]:
        assert (
            row["forward_simulation_terminal_residual"] < 1e-7
            and row["objective_difference"] < 1e-4
        )
    for row in read("results/reference/convex/evacuation/metrics.json")["rows"]:
        assert (
            max(
                row["mass_conservation_residual"],
                row["dynamics_residual"],
                row["capacity_violation"],
            )
            < 1e-8
        )
    for row in read("results/reference/convex/portfolio/metrics.json")["rows"]:
        assert row["duality_gap"] < 1e-6 and row["budget_residual"] < 1e-8
    spectral = read("results/reference/spectral/metrics.json")
    for row in spectral["qr"]:
        assert (
            row["converged"]
            and max(
                row["relative_eigen_residual"],
                row["orthogonality_residual"],
                row["maximum_eigenvalue_error"],
            )
            < 1e-10
        )
    for row in spectral["compression"]:
        np.testing.assert_allclose(
            row["raw_squared_error"], row["eckart_young_tail_sum"], rtol=1e-10, atol=1e-10
        )
    notebooks = list(Path("notebooks").glob("*.ipynb"))
    cells = 0
    if require_notebooks:
        assert len(notebooks) == 14
        for path in notebooks:
            notebook = nbformat.read(path, as_version=4)
            nbformat.validate(notebook)
            for cell in notebook.cells:
                if cell.cell_type == "code":
                    cells += 1
                    assert cell.execution_count is not None
                    assert not any(out.output_type == "error" for out in cell.outputs)
    figures = list(Path("assets/figures").glob("*.png"))
    for path in figures:
        assert path.with_suffix(".pdf").exists()
    result = dict(
        status="passed",
        reference_protocols=EXPECTED,
        numeric_artifacts=len(numeric),
        recommendation_candidate_evaluations=recommendation_candidates,
        selected_model_checkpoints=selected_models,
        routing_results=routes,
        notebooks=len(notebooks),
        executed_code_cells=cells,
        figure_pairs=len(figures),
    )
    write_json("docs/validation.json", result)
    print(result)
    return result
