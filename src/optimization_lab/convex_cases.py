"""Reconstructed convex applications with physical checks and independent witnesses."""

import json

import cvxpy as cp
import numpy as np
from scipy.optimize import linprog

from .barrier import speed_barrier, speed_certificate, speed_constraints
from .resources import config_path


def solve(problem, solver="CLARABEL"):
    options = {
        "CLARABEL": {"tol_gap_abs": 1e-9, "tol_gap_rel": 1e-9, "tol_feas": 1e-9, "max_iter": 500},
        "SCS": {"eps": 1e-6, "max_iters": 100000},
        "OSQP": {"eps_abs": 1e-8, "eps_rel": 1e-8, "max_iter": 100000, "polishing": True},
    }
    value = problem.solve(solver=solver, **options.get(solver, {}))
    if problem.status not in ["optimal", "optimal_inaccurate"] or not np.isfinite(value):
        raise RuntimeError(f"{solver}: {problem.status}")
    return {
        "solver": solver,
        "status": problem.status,
        "objective": float(value),
        "solver_seconds": problem.solver_stats.solve_time,
    }


def load_parameters(name):
    path = config_path(f"{name}_parameters.json")
    return json.loads(path.read_text())


def speed(parameters):
    d, k, c, g, h = speed_constraints(parameters)
    t = cp.Variable(len(d))
    constraint = g @ t <= h
    problem = cp.Problem(
        cp.Minimize(
            cp.sum(cp.multiply(k, cp.inv_pos(t))) + c * cp.sum(t) + parameters["b"] * d.sum()
        ),
        [constraint],
    )
    reference = solve(problem)
    certificate = speed_certificate(
        t.value, k, c, g, h, parameters["b"] * d.sum(), constraint.dual_value
    )
    owned = speed_barrier(parameters)
    return {
        "reference": reference,
        "certificate": certificate,
        "owned_certificate": owned["certificate"],
        "max_time_difference": float(np.max(abs(t.value - owned["time"]))),
        "arrays": {
            "reference_time": t.value,
            "barrier_time": owned["time"],
            "distance": d,
            "barrier_history": owned["history"],
            "prefix_min": parameters["tau_min"],
            "prefix_max": parameters["tau_max"],
            "speed_min": parameters["smin"],
            "speed_max": parameters["smax"],
        },
        "scope": "Full fuel includes constant b*sum(distance), omitted in the legacy optimization objective without changing its minimizer. Time transformation t=d/s uses positive distances and speeds.",
    }


def revenue(parameters):
    a = np.asarray(parameters["A"], float)
    p = np.asarray(parameters["p"], float).ravel()
    discount = np.asarray(parameters["p_disc"], float).ravel()
    q = np.asarray(parameters["q"], float).ravel()
    n = len(p)
    capacity = np.full(a.shape[0], 100.0)
    x = cp.Variable(n, nonneg=True)
    sales = cp.Variable(n)
    resource = a @ x <= capacity
    constraints = [
        resource,
        sales <= cp.multiply(p, x),
        sales <= p * q + cp.multiply(discount, x - q),
    ]
    problem = cp.Problem(cp.Maximize(cp.sum(sales)), constraints)
    reference = solve(problem)
    matrix = np.block(
        [
            [a, np.zeros((len(capacity), n))],
            [-np.diag(p), np.eye(n)],
            [-np.diag(discount), np.eye(n)],
        ]
    )
    rhs = np.r_[capacity, np.zeros(n), (p - discount) * q]
    objective = np.r_[np.zeros(n), -np.ones(n)]
    lp = linprog(
        objective,
        A_ub=matrix,
        b_ub=rhs,
        bounds=[(0, None)] * n + [(None, None)] * n,
        method="highs",
    )
    if not lp.success:
        raise RuntimeError(lp.message)
    true_sales = np.minimum(p * x.value, (p - discount) * q + discount * x.value)
    dual = -lp.ineqlin.marginals
    gap = float(rhs @ dual + lp.fun)
    perturbed = rhs.copy()
    perturbed[0] += 1e-4
    changed = linprog(
        objective,
        A_ub=matrix,
        b_ub=perturbed,
        bounds=[(0, None)] * n + [(None, None)] * n,
        method="highs",
    )
    return {
        "reference": reference,
        "highs_objective": float(-lp.fun),
        "solver_difference": abs(reference["objective"] + lp.fun),
        "duality_gap": gap,
        "resource_violation": float(max(0, np.max(a @ x.value - capacity))),
        "resource_shadow_prices": dual[: len(capacity)],
        "first_resource_finite_difference": float((-changed.fun + lp.fun) / 1e-4),
        "arrays": {
            "production": x.value,
            "sales": true_sales,
            "resource_use": a @ x.value,
            "capacity": capacity,
        },
        "scope": "Concave piecewise-linear revenue; resource duals are local marginal values, not a guarantee for arbitrary capacity changes.",
    }


def geometric_blend(parameters):
    w = np.asarray(parameters["W"], float)
    metrics = np.array([parameters["P"], parameters["D"], parameters["A"]], float)
    specs = np.array([parameters["P_spec"], parameters["D_spec"], parameters["A_spec"]], float)
    logs = np.log(metrics)
    bounds = np.log(specs)
    k = metrics.shape[1]
    rows = []
    arrays = {"original_designs": w, "metric_measurements": metrics, "specifications": specs}
    for objective in ["feasible", "min_power", "max_common_log_margin"]:
        if objective == "max_common_log_margin":
            cost = np.r_[np.zeros(k), -1.0]
            matrix = np.column_stack([logs, np.ones(3)])
            eq = np.r_[np.ones(k), 0.0][None]
            limits = [(0, None)] * k + [(None, None)]
        else:
            cost = np.zeros(k) if objective == "feasible" else logs[0]
            matrix = logs
            eq = np.ones((1, k))
            limits = [(0, None)] * k
        result = linprog(
            cost, A_ub=matrix, b_ub=bounds, A_eq=eq, b_eq=[1.0], bounds=limits, method="highs"
        )
        if not result.success:
            raise RuntimeError(result.message)
        alpha = result.x[:k]
        design = np.exp(np.log(w) @ alpha)
        bound = np.exp(logs @ alpha)
        rows.append(
            {
                "objective": objective,
                "simplex_residual": abs(alpha.sum() - 1.0),
                "max_relative_spec_violation": float(max(0, np.max(bound / specs - 1))),
                "common_log_margin": float(np.min(bounds - logs @ alpha)),
                "performance_upper_bounds": bound,
                "weights": alpha,
            }
        )
        arrays[objective + "_design"] = design
        arrays[objective + "_weights"] = alpha
    return {
        "rows": rows,
        "arrays": arrays,
        "scope": "Original feasibility uses geometric blends of six measured designs. Performance quantities are geometric upper bounds conditional on log-convex physical metrics, not independently measured predictions. Minimum power and common-log-margin are declared optimization extensions.",
    }


def nearest_psd(matrix):
    matrix = np.asarray(matrix, float)
    symmetric = 0.5 * (matrix + matrix.T)
    eigen, vectors = np.linalg.eigh(symmetric)
    projection = (vectors * np.maximum(eigen, 0.0)) @ vectors.T
    variable = cp.Variable(matrix.shape, symmetric=True)
    problem = cp.Problem(cp.Minimize(cp.sum_squares(variable - matrix)), [variable >> 0])
    reference = solve(problem)
    return {
        "reference": reference,
        "analytic_squared_distance": float(np.sum((projection - matrix) ** 2)),
        "solver_matrix_difference": float(np.linalg.norm(variable.value - projection)),
        "projection_min_eigenvalue": float(np.linalg.eigvalsh(projection).min()),
        "arrays": {
            "input": matrix,
            "projection": projection,
            "cvx_projection": variable.value,
            "eigenvalues_before": eigen,
        },
        "scope": "Closed convex cone of symmetric positive semidefinite matrices. Eigenvalue clipping follows symmetrization; strict positive definiteness is a different, open feasible set.",
    }


def regulator(parameters):
    a = np.asarray(parameters["A"], float)
    b = np.asarray(parameters["B"], float)
    initial = np.asarray(parameters["x_init"], float)
    horizon = parameters["T"]
    reach = np.column_stack(
        [np.linalg.matrix_power(a, horizon - 1 - t) @ b for t in range(horizon)]
    )
    target = -np.linalg.matrix_power(a, horizon) @ initial
    reference_vector = np.linalg.lstsq(reach, target, rcond=None)[0]
    least = reference_vector.reshape((b.shape[1], horizon), order="F")
    rows = []
    arrays = {
        "least_energy_control": least,
        "reachability_singular_values": np.linalg.svd(reach, compute_uv=False),
    }
    for name in ["frobenius", "group_sum", "group_peak", "element_l1"]:
        x = cp.Variable((len(initial), horizon + 1))
        u = cp.Variable((b.shape[1], horizon))
        objective = {
            "frobenius": cp.norm(u, "fro"),
            "group_sum": cp.sum(cp.norm(u, 2, axis=0)),
            "group_peak": cp.max(cp.norm(u, 2, axis=0)),
            "element_l1": cp.sum(cp.abs(u)),
        }[name]
        constraints = [x[:, 0] == initial, x[:, -1] == 0, x[:, 1:] == a @ x[:, :-1] + b @ u]
        problem = cp.Problem(cp.Minimize(objective), constraints)
        main = solve(problem)
        controls = u.value.copy()
        states = np.zeros((len(initial), horizon + 1))
        states[:, 0] = initial
        for t in range(horizon):
            states[:, t + 1] = a @ states[:, t] + b @ controls[:, t]
        residual = float(np.linalg.norm(states[:, -1], np.inf))
        if name == "frobenius":
            comparison = float(np.linalg.norm(least))
            kind = "independent SVD least-energy solution"
        elif name == "element_l1":
            n = reach.shape[1]
            matrix = np.block([[np.eye(n), -np.eye(n)], [-np.eye(n), -np.eye(n)]])
            lp = linprog(
                np.r_[np.zeros(n), np.ones(n)],
                A_ub=matrix,
                b_ub=np.zeros(2 * n),
                A_eq=np.column_stack([reach, np.zeros_like(reach)]),
                b_eq=target,
                bounds=[(None, None)] * n + [(0, None)] * n,
                method="highs",
            )
            if not lp.success:
                raise RuntimeError(lp.message)
            comparison = float(lp.fun)
            kind = "independent terminal-map HiGHS LP"
        else:
            comparison = solve(problem, "SCS")["objective"]
            kind = "SCS reference on identical declared cone model"
        rows.append(
            {
                "objective": name,
                "clarabel": main,
                "independent_reference_objective": comparison,
                "reference_type": kind,
                "objective_difference": abs(main["objective"] - comparison),
                "forward_simulation_terminal_residual": residual,
                "energy": float(np.sum(controls**2)),
                "max_action_norm": float(np.linalg.norm(controls, axis=0).max()),
            }
        )
        arrays[name + "_controls"] = controls
        arrays[name + "_states"] = states
    return {
        "rows": rows,
        "arrays": arrays,
        "scope": "Original x0=ones, xT=0, horizon100; four different norms. Squared energy shares the Frobenius minimizer but changes objective values and dual scaling. Nonunique norm optima are compared by feasibility and objective, not coefficient identity.",
    }


def evacuation(parameters):
    a = np.asarray(parameters["A"], float)
    capacity = np.asarray(parameters["Q"], float)
    flow_cap = np.asarray(parameters["F"], float)
    initial = np.asarray(parameters["q1"], float)
    r = np.asarray(parameters["r"], float)
    s = np.asarray(parameters["s"], float)
    edge_l1 = np.asarray(parameters["rtild"], float)
    edge_l2 = np.asarray(parameters["stild"], float)
    horizon = parameters["T"]
    if not np.allclose(a.sum(0), 0):
        raise ValueError("Incidence columns must conserve population")
    safe = (r == 0) & (s == 0)
    rows = []
    arrays = {"node_risk_weight": r, "node_capacity": capacity}
    for terminal in [False, True]:
        q = cp.Variable((len(capacity), horizon))
        f = cp.Variable((len(flow_cap), horizon - 1))
        objective = (
            cp.sum(cp.multiply(r[:, None], q))
            + cp.sum(cp.multiply(s[:, None], cp.square(q)))
            + cp.sum(cp.multiply(edge_l1[:, None], cp.abs(f)))
            + cp.sum(cp.multiply(edge_l2[:, None], cp.square(f)))
        )
        constraints = [
            q >= 0,
            q <= capacity[:, None],
            q[:, 0] == initial,
            q[:, 1:] == q[:, :-1] + a @ f,
            cp.abs(f) <= flow_cap[:, None],
        ]
        if terminal:
            constraints.append(cp.sum(q[safe, -1]) == initial.sum())
        problem = cp.Problem(cp.Minimize(objective), constraints)
        main = solve(problem)
        population = q.value.copy()
        flows = f.value.copy()
        secondary = solve(problem, "OSQP")
        rows.append(
            {
                "terminal_evacuation_constraint": terminal,
                "clarabel": main,
                "osqp": secondary,
                "objective_difference": abs(main["objective"] - secondary["objective"]),
                "mass_conservation_residual": float(np.max(abs(population.sum(0) - initial.sum()))),
                "dynamics_residual": float(
                    np.max(abs(population[:, 1:] - population[:, :-1] - a @ flows))
                ),
                "capacity_violation": float(
                    max(
                        0,
                        np.max(population - capacity[:, None]),
                        np.max(-population),
                        np.max(abs(flows) - flow_cap[:, None]),
                    )
                ),
                "terminal_unsafe_population": float(population[~safe, -1].sum()),
            }
        )
        prefix = "terminal" if terminal else "original"
        arrays[prefix + "_population"] = population
        arrays[prefix + "_flow"] = flows
    return {
        "rows": rows,
        "arrays": arrays,
        "scope": "Original risk minimization has no terminal evacuation requirement. Full safe-node terminal mass is a separately declared extension. Signed edges permit bidirectional flow and within-slot forwarding under balance equations; no extra old-population outflow rule is silently imposed.",
    }


def portfolio(parameters):
    mean = np.asarray(parameters["mu"], float)
    covariances = np.array([parameters[f"Sigma_{i + 1}"] for i in range(parameters["M"])], float)
    eigen = np.linalg.eigvalsh(covariances)
    if eigen.min() < -1e-10 or not np.allclose(covariances[:, 0], 0):
        raise ValueError("PSD covariances with the declared common riskless first asset required")
    rows = []
    arrays = {"scenario_covariances": covariances, "mean_returns": mean}
    for gamma in [0.1, parameters["gamma"], 10.0]:
        for risk_model in ["mean_covariance", "worst_scenario"]:
            w = cp.Variable(len(mean))
            t = cp.Variable()
            matrices = covariances if risk_model == "worst_scenario" else covariances.mean(0)[None]
            constraints = [cp.sum(w) == 1]
            risk_constraints = [cp.quad_form(w, matrix) <= t for matrix in matrices]
            problem = cp.Problem(cp.Maximize(mean @ w - gamma * t), constraints + risk_constraints)
            reference = solve(problem)
            weights = w.value.copy()
            weights[0] = 1 - weights[1:].sum()
            beta = np.maximum(
                np.array([constraint.dual_value for constraint in risk_constraints]).ravel(), 0.0
            )
            beta *= gamma / beta.sum()
            mixture = np.einsum("m,mij->ij", beta, matrices)[:, :]
            difference = mean[1:] - mean[0]
            dual = float(mean[0] + 0.25 * difference @ np.linalg.solve(mixture[1:, 1:], difference))
            risks = np.einsum("i,mij,j->m", weights, matrices, weights)
            primal = float(mean @ weights - gamma * risks.max())
            stationarity = mean - 2 * mixture @ weights - mean[0]
            rows.append(
                {
                    "gamma": gamma,
                    "risk_model": risk_model,
                    "reference": reference,
                    "primal_objective": primal,
                    "dual_objective": dual,
                    "duality_gap": dual - primal,
                    "budget_residual": abs(weights.sum() - 1.0),
                    "stationarity_residual": float(np.max(abs(stationarity))),
                    "scenario_dual_weights": beta,
                    "mean_return": float(mean @ weights),
                    "worst_variance_across_all_scenarios": float(
                        np.einsum("i,mij,j->m", weights, covariances, weights).max()
                    ),
                    "gross_exposure": float(np.abs(weights).sum()),
                }
            )
            arrays[f"{risk_model}_{gamma:g}_weights"] = weights
    return {
        "rows": rows,
        "arrays": arrays,
        "scope": "Original objective maximizes mean return minus gamma times worst scenario variance, with sum(weights)=1 and shorting allowed. Mean-covariance and gamma .1/10 are separate comparisons. The independently evaluated dual assumes a common riskless first asset and positive definite risky covariance mixture. These numerical course inputs are not out-of-sample investment performance.",
    }
