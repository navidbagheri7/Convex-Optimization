"""Independent mixed-integer TSP formulation for small exact comparisons."""

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import lil_matrix

from .routing import tour_length


def mtz_tsp(distance, seconds=30):
    n = len(distance)
    edges = [(i, j) for i in range(n) for j in range(n) if i != j]
    lookup = {edge: index for index, edge in enumerate(edges)}
    m = len(edges)
    matrix = lil_matrix((2 * n + (n - 1) * (n - 2), m + n - 1))
    lower = np.full(matrix.shape[0], -np.inf)
    upper = np.empty(matrix.shape[0])
    for i in range(n):
        for j in range(n):
            if i != j:
                matrix[i, lookup[i, j]] = 1
                matrix[n + i, lookup[j, i]] = 1
        lower[i] = lower[n + i] = upper[i] = upper[n + i] = 1
    row = 2 * n
    for i in range(1, n):
        for j in range(1, n):
            if i != j:
                matrix[row, lookup[i, j]] = n
                matrix[row, m + i - 1] = 1
                matrix[row, m + j - 1] = -1
                upper[row] = n - 1
                row += 1
    cost = np.r_[[distance[i, j] for i, j in edges], np.zeros(n - 1)]
    bounds = Bounds(np.r_[np.zeros(m), np.ones(n - 1)], np.r_[np.ones(m), np.full(n - 1, n - 1)])
    solution = milp(
        cost,
        integrality=np.r_[np.ones(m), np.zeros(n - 1)],
        bounds=bounds,
        constraints=LinearConstraint(matrix.tocsr(), lower, upper),
        options=dict(time_limit=seconds, mip_rel_gap=0.0),
    )
    if solution.x is None:
        raise RuntimeError("No feasible MILP solution")
    successor = {}
    for (i, j), value in zip(edges, solution.x[:m]):
        if value > 0.5:
            successor[i] = j
    route = [0]
    while successor[route[-1]] != 0:
        route.append(successor[route[-1]])
        if len(route) > n:
            raise ValueError("Invalid MILP cycle")
    route = np.array(route)
    value = tour_length(distance, route)
    return dict(
        tour=route,
        length=value,
        solver_status=int(solution.status),
        message=solution.message,
        mip_dual_bound=float(solution.mip_dual_bound),
        mip_gap=float(solution.mip_gap),
        proven_optimal=bool(solution.status == 0 and solution.mip_gap == 0),
        objective_rounding_residual=float(abs(value - solution.fun)),
    )
