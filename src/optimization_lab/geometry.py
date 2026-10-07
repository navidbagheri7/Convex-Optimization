"""Four explicitly scoped transformed convex feasibility domains."""

import cvxpy as cp
import numpy as np

from .convex_cases import solve


def transformed_domains():
    results = []
    x, y = cp.Variable(), cp.Variable()
    programs = [
        ("simplex", [x >= 0, y >= 0, x + y <= 1], cp.sum_squares(cp.hstack([x - 0.3, y - 0.4]))),
        (
            "nonnegative_order",
            [x >= 0, y >= 0, x - y >= 0],
            cp.sum_squares(cp.hstack([x - 1, y - 0.5])),
        ),
        (
            "perspective_quadratic",
            [y >= 0, cp.quad_over_lin(x, y) <= x - 2 * cp.square(y) + 5],
            cp.sum_squares(cp.hstack([x - 1, y - 1])),
        ),
    ]
    for name, constraints, objective in programs:
        problem = cp.Problem(cp.Minimize(objective), constraints)
        if not problem.is_dcp():
            raise ValueError("Invalid convex composition")
        reference = solve(problem)
        results.append(
            dict(
                domain=name,
                reference=reference,
                witness=[float(x.value), float(y.value)],
                maximum_violation=float(max(np.max(c.violation()) for c in constraints)),
            )
        )
    # Retain the full square-root composition with a free transformed t.
    u, t, s = cp.Variable(), cp.Variable(), cp.Variable()
    constraints = [u >= 0, s >= 0, u + t <= s + cp.sqrt(u - cp.square(t))]
    problem = cp.Problem(cp.Minimize(cp.sum_squares(cp.hstack([u - 1, t, s - 0.5]))), constraints)
    assert problem.is_dcp()
    reference = solve(problem)
    results.append(
        dict(
            domain="concave_sqrt_hypograph",
            reference=reference,
            witness=[float(u.value), float(t.value), float(s.value)],
            maximum_violation=float(max(np.max(c.violation()) for c in constraints)),
        )
    )
    return dict(
        rows=results,
        scope="Authored DCP-domain witnesses for the supplied transformed forms. Positive perspective denominator and nonnegative square-root radicand are part of the domains. This does not assert a unique inverse transformation to an ambiguously specified original nonconvex program.",
    )
