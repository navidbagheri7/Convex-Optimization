import numpy as np

from optimization_lab.memetic import hybrid
from optimization_lab.routing import tour_length


def test_hybrid_charges_initialization_and_improvement():
    rng = np.random.default_rng(3)
    xy = rng.normal(size=(12, 2)) * 100
    d = np.floor(np.linalg.norm(xy[:, None] - xy[None], axis=2) + 0.5).astype(np.int64)
    for method in ["ox", "cx"]:
        result = hybrid(d, 0, 100000, method, population_size=8)
        assert result["length"] == tour_length(d, result["tour"])
        assert result["edge_work"] <= 100000
        assert np.all(np.diff(result["history"][:, 1]) < 0)
