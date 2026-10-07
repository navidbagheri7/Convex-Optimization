import numpy as np

from optimization_lab.barrier import speed_barrier


def test_reciprocal_fuel_oracle_and_dual_gap():
    p = {
        "d": [1.0, 2.0],
        "a": 1.0,
        "b": 6.0,
        "c": 4.0,
        "smin": [0.5, 0.5],
        "smax": [4.0, 4.0],
        "tau_min": [0.1, 0.2],
        "tau_max": [10.0, 20.0],
    }
    result = speed_barrier(p, 1e8)
    assert np.allclose(result["speed"], [2.0, 2.0], atol=1e-5)
    assert abs(result["certificate"]["objective"] - 30.0) < 1e-6
    assert -1e-10 < result["certificate"]["gap"] < 1e-6
    assert result["certificate"]["constraint_violation"] == 0
