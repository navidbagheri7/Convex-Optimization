"""Owned TSP search and exact integer 1-tree certificates; no optimal tours loaded."""

import gzip
import time
from pathlib import Path

import numpy as np


def load_tsplib(path):
    content = Path(path).read_bytes()
    if content[:2] == b"\x1f\x8b":
        content = gzip.decompress(content)
    text = content.decode("ascii")
    header, body = text.split("NODE_COORD_SECTION", 1)
    metadata = dict(line.split(":", 1) for line in header.splitlines() if ":" in line)
    metadata = {k.strip(): v.strip() for k, v in metadata.items()}
    if metadata.get("EDGE_WEIGHT_TYPE") != "EUC_2D":
        raise ValueError("Only TSPLIB EUC_2D is supported")
    rows = []
    for line in body.splitlines():
        if line.strip() == "EOF":
            break
        if line.strip():
            rows.append([float(x) for x in line.split()])
    values = np.asarray(rows)
    n = int(metadata["DIMENSION"])
    if values.shape != (n, 3) or not np.array_equal(values[:, 0], np.arange(1, n + 1)):
        raise ValueError("Ordered, unique coordinates required")
    xy = values[:, 1:]
    distances = np.floor(np.linalg.norm(xy[:, None] - xy[None], axis=2) + 0.5).astype(np.int64)
    return metadata["NAME"], xy, distances


def tour_length(distance, tour):
    tour = np.asarray(tour)
    if not np.issubdtype(tour.dtype, np.integer) or not np.array_equal(
        np.sort(tour), np.arange(len(distance))
    ):
        raise ValueError("Tour must visit each node exactly once")
    return int(distance[tour, np.roll(tour, -1)].sum())


class Budget:
    def __init__(self, distance, limit):
        self.distance = distance
        self.limit = int(limit)
        self.work = 0
        self.objectives = 0
        self.start = time.perf_counter()
        self.best = np.iinfo(np.int64).max
        self.tour = None
        self.history = []

    def room(self, work):
        return self.work + work <= self.limit

    def evaluate(self, tour):
        if not self.room(len(tour)):
            raise ValueError("Objective exceeds declared edge-work budget")
        self.work += len(tour)
        self.objectives += 1
        value = tour_length(self.distance, tour)
        self.last_value = value
        if value < self.best:
            self.best = value
            self.tour = np.copy(tour)
            self.history.append((self.work, value, time.perf_counter() - self.start))
        return value


def nearest_neighbor(budget, start=0):
    n = len(budget.distance)
    remaining = set(range(n)) - {start}
    tour = [start]
    while remaining:
        budget.work += len(remaining)
        nxt = min(remaining, key=lambda j: (budget.distance[tour[-1], j], j))
        tour.append(nxt)
        remaining.remove(nxt)
    tour = np.array(tour)
    budget.evaluate(tour)
    return tour


def two_opt(budget, tour, max_passes=100):
    tour = np.copy(tour)
    n = len(tour)
    for _ in range(max_passes):
        improved = False
        for i in range(n - 2):
            for j in range(i + 2, n):
                if i == 0 and j == n - 1:
                    continue
                if not budget.room(4 + n):
                    return tour
                budget.work += 4
                a, b, c, d = tour[i], tour[i + 1], tour[j], tour[(j + 1) % n]
                delta = (
                    budget.distance[a, c]
                    + budget.distance[b, d]
                    - budget.distance[a, b]
                    - budget.distance[c, d]
                )
                if delta < 0:
                    tour[i + 1 : j + 1] = tour[i + 1 : j + 1][::-1]
                    budget.evaluate(tour)
                    improved = True
        if not improved:
            break
    return tour


def mutate(tour, rng, neighborhood):
    result = np.copy(tour)
    i, j = sorted(rng.choice(len(tour), 2, replace=False))
    if neighborhood == "swap":
        result[i], result[j] = result[j], result[i]
    elif neighborhood == "reverse":
        result[i : j + 1] = result[i : j + 1][::-1]
    elif neighborhood == "insert":
        result = np.insert(np.delete(result, i), j, result[i])
    else:
        raise ValueError("Unknown neighborhood")
    return result


def order_crossover(a, b, rng):
    lo, hi = sorted(rng.choice(len(a), 2, replace=False))
    hi += 1
    result = np.full(len(a), -1, dtype=int)
    result[lo:hi] = a[lo:hi]
    inherited = set(a[lo:hi])
    remaining = [int(x) for x in np.roll(b, -hi) if x not in inherited]
    places = list(range(hi, len(a))) + list(range(lo))
    result[places] = remaining
    return result


def cycle_crossover(a, b, rng=None):
    result = np.empty_like(a)
    inverse = np.argsort(a)
    unvisited = set(range(len(a)))
    cycle = 0
    while unvisited:
        index = min(unvisited)
        while index in unvisited:
            result[index] = a[index] if cycle % 2 == 0 else b[index]
            unvisited.remove(index)
            index = int(inverse[b[index]])
        cycle += 1
    return result


def annealing(budget, rng, neighborhood):
    tour = nearest_neighbor(budget, int(rng.integers(len(budget.distance))))
    value = budget.last_value
    initial = max(1.0, 2 * np.median(budget.distance[budget.distance > 0]))
    while budget.room(len(tour)):
        fraction = budget.work / budget.limit
        temperature = initial * (1 / initial) ** fraction
        candidate = mutate(tour, rng, neighborhood)
        cost = budget.evaluate(candidate)
        if cost <= value or rng.random() < np.exp((value - cost) / temperature):
            tour, value = candidate, cost


def genetic(budget, rng, crossover, mutation, population_size=64):
    population = []
    costs = []
    for _ in range(population_size):
        route = rng.permutation(len(budget.distance))
        population.append(route)
        costs.append(budget.evaluate(route))
    cross = order_crossover if crossover == "ox" else cycle_crossover
    while budget.room(len(budget.distance)):
        parents = []
        for _ in range(2):
            contenders = rng.choice(population_size, 3, replace=False)
            parents.append(population[min(contenders, key=lambda x: costs[x])])
        child = cross(*parents, rng)
        if rng.random() < 0.3:
            child = mutate(child, rng, mutation)
        value = budget.evaluate(child)
        worst = int(np.argmax(costs))
        if value <= costs[worst]:
            population[worst], costs[worst] = child, value
        # Steady-state elitism can collapse diversity: explicit 2% immigration.
        if rng.random() < 0.02 and budget.room(len(child)):
            child = rng.permutation(len(child))
            population[worst], costs[worst] = child, budget.evaluate(child)


def ant_colony(budget, rng, variant, ants=16):
    n = len(budget.distance)
    nearest_neighbor(budget)
    baseline = budget.last_value
    tau0 = 1 / (n * baseline)
    trails = np.full((n, n), tau0)
    rho = 0.1
    construction_work = n * (n - 1) // 2 + n
    while budget.room(construction_work):
        iteration = []
        for _ in range(ants):
            if not budget.room(construction_work):
                break
            remaining = np.ones(n, bool)
            route = [int(rng.integers(n))]
            remaining[route[0]] = False
            while remaining.any():
                current = route[-1]
                choices = np.flatnonzero(remaining)
                budget.work += len(choices)
                weight = (
                    trails[current, choices] / np.maximum(1, budget.distance[current, choices]) ** 3
                )
                if variant != "as" and rng.random() < 0.9:
                    nxt = int(choices[np.argmax(weight)])
                else:
                    nxt = int(rng.choice(choices, p=weight / weight.sum()))
                if variant != "as":
                    trails[current, nxt] = trails[nxt, current] = (
                        0.9 * trails[current, nxt] + 0.1 * tau0
                    )
                route.append(nxt)
                remaining[nxt] = False
            route = np.array(route)
            if variant != "as":
                a, b = route[-1], route[0]
                trails[a, b] = trails[b, a] = 0.9 * trails[a, b] + 0.1 * tau0
            value = budget.evaluate(route)
            if variant == "acs_2opt":
                route = two_opt(budget, route, max_passes=2)
                # Last accepted route was already evaluated, no hidden optimization.
                value = budget.last_value
            iteration.append((route, value))
        if variant == "as":
            trails *= 1 - rho
            for route, value in iteration:
                a, b = route, np.roll(route, -1)
                trails[a, b] += 1 / value
                trails[b, a] += 1 / value
        else:
            a, b = budget.tour, np.roll(budget.tour, -1)
            updated = (1 - rho) * trails[a, b] + rho / budget.best
            trails[a, b] = trails[b, a] = updated
        trails = np.maximum(trails, np.finfo(float).tiny)


def search(distance, method, seed, edge_budget):
    n = len(distance)
    if edge_budget < n * (n - 1) // 2 + n * 64:
        raise ValueError("Budget must cover initialization")
    budget = Budget(distance, edge_budget)
    rng = np.random.default_rng(seed)
    if method == "nn_2opt":
        two_opt(budget, nearest_neighbor(budget, int(rng.integers(n))))
    elif method.startswith("sa_"):
        annealing(budget, rng, method[3:])
    elif method.startswith("ga_"):
        _, crossover, mutation = method.split("_")
        genetic(budget, rng, crossover, mutation)
    elif method in ["as", "acs", "acs_2opt"]:
        ant_colony(budget, rng, method)
    else:
        raise ValueError("Unknown owned method")
    return dict(
        tour=budget.tour,
        length=budget.best,
        history=np.array(budget.history),
        edge_work=budget.work,
        objective_evaluations=budget.objectives,
        elapsed_seconds=time.perf_counter() - budget.start,
    )


def integer_one_tree(distance, potentials):
    distance = np.asarray(distance)
    pi = np.asarray(potentials)
    n = len(distance)
    if (
        n < 3
        or distance.shape != (n, n)
        or pi.shape != (n,)
        or not np.issubdtype(distance.dtype, np.integer)
        or not np.issubdtype(pi.dtype, np.integer)
    ):
        raise ValueError("Integer distances and integer node potentials required")
    if (
        not np.array_equal(distance, distance.T)
        or np.any(distance < 0)
        or np.any(np.diag(distance))
    ):
        raise ValueError("Symmetric nonnegative distances with zero diagonal required")
    if max(max(abs(int(value)) for value in pi), int(np.max(distance))) * (8 * n) >= 2**62:
        raise ValueError("Certificate arithmetic exceeds conservative int64 range")
    modified = distance.astype(np.int64) + pi[:, None] + pi[None, :]
    used = np.zeros(n, bool)
    used[0] = used[1] = True
    minimum = modified[1].copy()
    parent = np.ones(n, int)
    edges = []
    for _ in range(n - 2):
        choices = np.flatnonzero(~used)
        nxt = int(choices[np.argmin(minimum[choices])])
        edges.append((int(parent[nxt]), nxt))
        used[nxt] = True
        for j in np.flatnonzero(~used):
            if modified[nxt, j] < minimum[j]:
                minimum[j], parent[j] = modified[nxt, j], nxt
    neighbors = np.argsort(modified[0, 1:], kind="stable")[:2] + 1
    edges.extend((0, int(j)) for j in neighbors)
    degree = np.bincount(np.asarray(edges).ravel(), minlength=n)
    bound = sum(int(modified[a, b]) for a, b in edges) - 2 * sum(map(int, pi))
    return dict(
        bound=bound,
        degree=degree,
        edges=np.array(edges),
        potentials=pi.copy(),
        is_hamiltonian=bool(np.all(degree == 2)),
    )


def one_tree_bound(distance, upper, steps=1500):
    pi = np.zeros(len(distance))
    best = None
    history = []
    scale = 2.0
    stale = 0
    for step in range(steps):
        result = integer_one_tree(distance, np.rint(pi).astype(np.int64))
        if result["bound"] > upper:
            raise ValueError("Lower bound exceeds feasible incumbent")
        if best is None or result["bound"] > best["bound"]:
            best = result
            stale = 0
            history.append((step, result["bound"]))
        else:
            stale += 1
        if result["is_hamiltonian"] or result["bound"] == upper:
            break
        direction = result["degree"] - 2
        pi += scale * (upper - result["bound"]) / (direction @ direction) * direction
        if stale >= 100:
            scale *= 0.5
            stale = 0
    best["history"] = np.array(history)
    best["iterations"] = step + 1
    return best


def held_karp(distance):
    n = len(distance)
    if not 2 <= n <= 18:
        raise ValueError("Exact dynamic programming limited to2..18 nodes")
    dp = {(1 << (j - 1), j): (int(distance[0, j]), 0) for j in range(1, n)}
    for mask in range(1, 1 << (n - 1)):
        for j in range(1, n):
            if not mask & (1 << (j - 1)):
                continue
            previous = mask ^ (1 << (j - 1))
            if previous == 0:
                continue
            dp[mask, j] = min(
                (dp[previous, k][0] + int(distance[k, j]), k)
                for k in range(1, n)
                if previous & (1 << (k - 1))
            )
    mask = (1 << (n - 1)) - 1
    value, last = min((dp[mask, j][0] + int(distance[j, 0]), j) for j in range(1, n))
    route = []
    while last:
        route.append(last)
        _, parent = dp[mask, last]
        mask ^= 1 << (last - 1)
        last = parent
    tour = np.array([0] + route[::-1])
    assert tour_length(distance, tour) == value
    return dict(tour=tour, length=value, states=len(dp))


def ortools_search(distance, start, seconds=5):
    from ortools.constraint_solver import pywrapcp, routing_enums_pb2

    manager = pywrapcp.RoutingIndexManager(len(distance), 1, int(start))
    routing = pywrapcp.RoutingModel(manager)
    callback_calls = [0]

    def callback(a, b):
        callback_calls[0] += 1
        return int(distance[manager.IndexToNode(a), manager.IndexToNode(b)])

    index = routing.RegisterTransitCallback(callback)
    routing.SetArcCostEvaluatorOfAllVehicles(index)
    params = pywrapcp.DefaultRoutingSearchParameters()
    params.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
    params.local_search_metaheuristic = (
        routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
    )
    params.time_limit.FromMilliseconds(int(seconds * 1000))
    beginning = time.perf_counter()
    solution = routing.SolveWithParameters(params)
    if solution is None:
        raise RuntimeError("OR-Tools found no feasible tour")
    cursor = routing.Start(0)
    route = []
    while not routing.IsEnd(cursor):
        route.append(manager.IndexToNode(cursor))
        cursor = solution.Value(routing.NextVar(cursor))
    route = np.array(route)
    return dict(
        tour=route,
        length=tour_length(distance, route),
        elapsed_seconds=time.perf_counter() - beginning,
        callback_evaluations=callback_calls[0],
        solver_status=int(routing.status()),
    )
