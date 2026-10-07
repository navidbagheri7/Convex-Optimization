import numpy as np

from optimization_lab.first_order import lasso, nuclear_completion, poisson_prox, poisson_smooth


def test_lasso_identity_oracle_and_dual_feasibility():
    b = np.array([3.0, -2.0, 0.25])
    for method in ["ista", "fista", "admm"]:
        result = lasso(np.eye(3), b, 1.0, method, steps=1000)
        assert np.allclose(result["x"], [2.0, -1.0, 0.0], atol=1e-6)
        assert result["certificate"]["gap"] >= -1e-12
        assert result["certificate"]["dual_feasibility"] <= 1e-10
        assert result["certificate"]["gap"] < 1e-6


def test_nuclear_diagonal_oracle_and_masked_missing_values():
    result = nuclear_completion(np.diag([4.0, 2.0, 0.2]), np.ones((3, 3), bool), 1.0)
    assert np.allclose(result["x"], np.diag([3.0, 1.0, 0.0]), atol=1e-7)
    assert result["certificate"]["gap"] < 1e-7
    y = np.array([[4.0, np.nan], [np.nan, 2.0]])
    masked = nuclear_completion(y, np.isfinite(y), 1.0)
    assert np.isfinite(masked["x"]).all()


def test_poisson_boundary_and_cancellation_stability():
    assert np.array_equal(poisson_prox(np.array([0.5, 3.0]), np.zeros(2), 1.0), [0.0, 2.0])
    assert poisson_prox(np.array([-1e12]), np.array([1.0]), 1.0)[0] > 0
    result = poisson_smooth([0, 4, 0, 2], 0.0)
    assert np.allclose(result["rate"], [0, 4, 0, 2], atol=1e-5)
    assert result["kkt_residual"] < 1e-6


def test_nonfinite_parameters_rejected_and_scalar_prox_boundary():
    import pytest

    assert poisson_prox(0.5, 0.0, 1.0) == 0.0
    assert poisson_prox(-1e12, 1.0, 1.0) > 0.0
    for value in [np.nan, np.inf]:
        with pytest.raises(ValueError):
            lasso(np.eye(2), np.ones(2), value)
        with pytest.raises(ValueError):
            nuclear_completion(np.eye(2), np.ones((2, 2), bool), value)
        with pytest.raises(ValueError):
            poisson_prox(1.0, 2.0, value)
        with pytest.raises(ValueError):
            poisson_smooth([1, 2], value)
