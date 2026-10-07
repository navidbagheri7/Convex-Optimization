import numpy as np
import pytest

from optimization_lab.deployment import load_factors, predict_ratings


def test_portable_inference_and_readonly_validated_shapes(tmp_path):
    path = tmp_path / "factors.npz"
    np.savez(
        path, mean=3.0, user=[[1.0, 2.0]], item=[[2.0, 1.0]], user_bias=[0.2], item_bias=[-0.1]
    )
    values = predict_ratings(path, np.array([0, -1, 0]), np.array([0, -1, -1]), clip=False)
    np.testing.assert_allclose(values, [7.1, 3.0, 3.2])
    np.testing.assert_allclose(predict_ratings(path, np.array([0]), np.array([0])), [5.0])
    assert not load_factors(path)["user"].flags.writeable
    np.savez(
        path, mean=3.0, user=[[1.0, 2.0]], item=[[2.0, 1.0]], user_bias=[0.2, 0.3], item_bias=[-0.1]
    )
    with pytest.raises(ValueError):
        load_factors(path)
