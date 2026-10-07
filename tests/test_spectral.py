import numpy as np

from optimization_lab.spectral import channel_svd, nonsymmetric_schur, symmetric_qr


def test_repeated_eigenvalues_and_reconstruction():
    rng = np.random.default_rng(42)
    q, _ = np.linalg.qr(rng.normal(size=(8, 8)))
    values = np.array([-2.0, 0.0, 0.0, 1.0, 2.0, 2.0, 5.0, 9.0])
    a = (q * values) @ q.T
    result = symmetric_qr(a)
    assert result["converged"] and len(result["values"]) == 8
    assert np.allclose(result["values"], values, atol=1e-10)
    assert result["relative_eigen_residual"] < 1e-10
    assert result["orthogonality_residual"] < 1e-10


def test_schur_vectors_are_not_claimed_as_eigenvectors():
    a = np.array([[1.0, 100.0], [0.0, 2.0]])
    result = nonsymmetric_schur(a)
    assert result["relative_reconstruction_residual"] < 1e-12
    assert np.linalg.norm(a @ result["vectors"] - result["vectors"] * result["values"]) > 10


def test_truncated_svd_has_optimal_tail_error():
    image = np.random.default_rng(0).random((12, 10, 3))
    rows, _ = channel_svd(image, [1, 4, 10])
    for row in rows:
        assert abs(row["raw_squared_error"] - row["eckart_young_tail_sum"]) < 1e-10
