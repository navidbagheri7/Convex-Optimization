# Numerical Optimization Lab

From convex certificates to recommendation and combinatorial search: an English, executable research portfolio by **Navid Bagheri-Shouraki**.

This repository reconstructs the main optimization course projects and extends them into reproducible comparisons. It connects each numerical result to its objective, assumptions, data roles and evidence. A solver's success flag is checked against independent calculations; a good training objective is checked against heldout performance; a heuristic route is checked against a genuine lower bound or a published optimum.

![Routing comparison](assets/figures/routing_comparison.png)
![Recommendation comparison](assets/figures/recommendation_comparison.png)

## What is implemented

| Study | Owned implementation | Independent reference / evidence |
|---|---|---|
| Sparse regression | ISTA, FISTA, ADMM | CVXPY, sklearn with matched normalization, feasible Lasso dual gap |
| Low-rank completion | Exact-SVD proximal ISTA / FISTA | Nuclear-norm conic reference, spectral dual feasibility |
| Poisson intensity estimation | Boundary-correct likelihood proximal smoothing | Exponential-cone reference, projected KKT, validation / synthetic test days |
| Speed scheduling | Feasible-start Newton barrier | Reciprocal convex formulation and a feasible dual certificate |
| Convex applications | Explicit revenue, geometric blending, PSD projection, regulator, evacuation, robust-risk models | HiGHS primal/dual LP, analytic eigen clipping, controllability SVD, independent solver comparisons |
| Spectral foundations | Householder reduction, shifted and deflated symmetric QR, SVD compression, PCA | Repeated eigenvalue checks, Schur counterexample, Eckart–Young error identity, sklearn subspace comparison |
| MovieLens recommendation | Biased degree-regularized ALS, minibatch SGD, Adam | Official fixed test split, validation-only search, three fitting seeds, cold-item fallback, user bootstrap |
| Routing | Three annealing neighborhoods, OX / CX genetic variants, Ant System, ACS, ACS + 2-opt, NN + 2-opt | OR-Tools guided local search, TSPLIB integer distances, exact integer 1-tree bounds, Held–Karp / MTZ MILP |

The original main projects are **MovieLens matrix factorization and Berlin52 / ch150 routing**, rather than only synthetic examples. The synthetic studies supply mathematical oracles and controlled tests. [Source audit](docs/source_audit.md), [theory](docs/theory.md), [protocol](docs/protocol.md), [measured results](docs/results.md) and [limitations](docs/model_cards.md) explain the connection.

## Start locally

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[recommendation,routing,notebooks,dev]'
pytest -q --tb=short
optimization-lab convex --quick
optimization-lab first-order --quick
optimization-lab spectral --quick
```

Quick runs are smoke checks and write to `results/smoke`; they are not the published benchmarks. The installed CLI carries its default numerical configurations and licensed example photo, so these commands also work outside the checkout. Full references, figures and executed notebooks are included here. [Reproduction](docs/reproduction.md) gives the exact budgets, GPU commands and cache layout.

For the two main projects, set a personal cache rather than storing datasets inside Git:

```bash
export OPTIMIZATION_DATA_ROOT=/data/navid.bagheri-shouraki/ML_data
optimization-lab data --dataset movielens --data-root "$OPTIMIZATION_DATA_ROOT"
optimization-lab data --dataset routing --data-root "$OPTIMIZATION_DATA_ROOT"
optimization-lab recommendation --data-root "$OPTIMIZATION_DATA_ROOT" --seed 0 --device cuda:0
optimization-lab routing --instance-root "$OPTIMIZATION_DATA_ROOT/optimization/routing"
```

CPU recommendation fitting is supported. Repeat recommendation with seeds 1 and 2 for the full comparison. The route study runs all three seeds itself. `optimization-lab report` rebuilds the reports and figures from complete reference evidence; `optimization-lab validate` audits that evidence without needing raw ratings or private checkpoints.

## Read the experiments

The [notebooks](notebooks/) are authored English walkthroughs with real executed outputs. They contain small analytical checks, measured tables and figures; they do not trigger the full expensive training campaign when opened. Numeric figure inputs live in `assets/source_data`, complete protocols in `results/reference`, and the numerical package in `src/optimization_lab`.

The low-regularization MovieLens experiment is retained alongside the validation-search extension. Some genetic and annealing variants remain well behind the stronger routing methods. The unconstrained original portfolio permits substantial short exposure. These are measured limitations, not omitted failures. Three seeds characterize this benchmark, not broad scientific superiority or production readiness.

## Publication and attribution

The new code and authored documentation use MIT. The example photograph and Iris data have separate attribution in [third-party notices](docs/third_party.md). MovieLens raw ratings and trained factor checkpoints stay in the local cache / ignored checkpoint directories. Course PDFs, scans, books and other students' code remain private under `legacy/originals`; only the detected-content hash manifest is public. No GitHub upload is performed by the preparation workflow.

The package is a tested **reference release**, with its measured stack recorded in [requirements-reference.txt](requirements-reference.txt). It does not claim that those pins are the newest releases. Cite this repository through [CITATION.cff](CITATION.cff) and the datasets / algorithms listed in [references](docs/references.md).
