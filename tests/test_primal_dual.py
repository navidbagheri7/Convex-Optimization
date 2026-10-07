"""Independent adjoint, cone-oracle and feasible-dual checks."""

import numpy as np
import pytest

from optimization_lab.primal_dual import adjoint, certificate, denoise, gradient
from optimization_lab.primal_dual_study import conic_reference


def test_periodic_adjoint_and_constant_nullspace():
    rng = np.random.default_rng(12)
    x = rng.normal(size=(7, 9))
    p = rng.normal(size=(2, 7, 9))
    np.testing.assert_allclose(np.sum(gradient(x) * p), np.sum(x * adjoint(p)), atol=1e-12)
    assert np.array_equal(gradient(np.ones((7, 9))), np.zeros((2, 7, 9)))


def test_all_solvers_against_independent_cone_model():
    b = np.random.default_rng(20).normal(size=(8, 7))
    lam = 0.18
    oracle, x = conic_reference(b, lam)
    for method in ["pdhg", "accelerated_pdhg", "dual_fista", "admm"]:
        result = denoise(b, lam, method, steps=12000, gap_per_pixel=2e-9, record_every=20)
        cert = result["certificate"]
        assert result["converged"]
        assert -1e-10 <= cert["gap"] <= b.size * 2e-9
        assert cert["dual_feasibility"] < 1e-12
        assert abs(cert["primal"] - oracle["objective"]) < 1e-6
        np.testing.assert_allclose(result["x"], x, atol=5e-4, rtol=0)
        assert result["snapshot_iterations"][0] == 0
        assert result["snapshot_iterations"][-1] == result["iterations"]
        assert np.isfinite(result["history"]).all()


def test_certificate_projects_unfeasible_dual_and_invalid_inputs_rejected():
    b = np.random.default_rng(3).normal(size=(5, 6))
    cert = certificate(b, b, np.full((2, 5, 6), 1e8), 0.2)
    assert cert["dual_feasibility"] <= 1e-12
    assert cert["gap"] >= -1e-12
    for lam in [0, -1, np.inf, np.nan]:
        with pytest.raises(ValueError):
            denoise(b, lam)
    with pytest.raises(ValueError):
        denoise(np.array([[np.nan, 1], [0, 1]]), 0.1)
