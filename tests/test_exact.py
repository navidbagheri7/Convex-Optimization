import numpy as np

from optimization_lab.exact import mtz_tsp
from optimization_lab.routing import held_karp


def test_milp_and_dynamic_programming_independent_exact_agreement():
    rng = np.random.default_rng(62)
    xy = rng.normal(size=(7, 2)) * 100
    d = np.floor(np.linalg.norm(xy[:, None] - xy[None], axis=2) + 0.5).astype(np.int64)
    result = mtz_tsp(d)
    assert result["proven_optimal"]
    assert result["length"] == held_karp(d)["length"]
    assert abs(result["length"] - result["mip_dual_bound"]) < 1e-7
