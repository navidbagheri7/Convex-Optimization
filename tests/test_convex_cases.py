import numpy as np

from optimization_lab.convex_cases import nearest_psd, portfolio, regulator


def test_nonsymmetric_psd_projection_symmetrizes_and_clips():
    result = nearest_psd(np.array([[1.0, 3.0], [-1.0, -2.0]]))
    projected = result["arrays"]["projection"]
    assert np.allclose(projected, projected.T)
    assert np.linalg.eigvalsh(projected).min() > -1e-12
    assert abs(result["reference"]["objective"] - result["analytic_squared_distance"]) < 1e-6


def test_lqr_forward_witness_matches_terminal_map():
    p = {"A": [[1.0]], "B": [[1.0]], "x_init": [2.0], "T": 4}
    result = regulator(p)
    assert np.allclose(result["arrays"]["least_energy_control"], -0.5)
    for row in result["rows"]:
        assert row["forward_simulation_terminal_residual"] < 1e-6
        assert row["objective_difference"] < 1e-4


def test_riskless_portfolio_dual_oracle():
    p = {"mu": [0.0, 0.2], "Sigma_1": [[0.0, 0.0], [0.0, 1.0]], "M": 1, "gamma": 1.0}
    result = portfolio(p)
    primary = next(
        r for r in result["rows"] if r["gamma"] == 1.0 and r["risk_model"] == "worst_scenario"
    )
    assert abs(primary["primal_objective"] - 0.01) < 1e-8
    assert primary["duality_gap"] < 1e-8
