"""Authored English numerical walkthroughs; execute separately before release."""

from pathlib import Path

import nbformat as nbf

SETUP = """from pathlib import Path
import sys, json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import Image, display
ROOT=Path.cwd()
if ROOT.name=='notebooks': ROOT=ROOT.parent
sys.path.insert(0,str(ROOT/'src'))
def read(path): return json.loads((ROOT/path).read_text())
def figure(name): display(Image(filename=str(ROOT/'assets/figures'/f'{name}.png')))
"""
STUDIES = [
    (
        "01_lasso",
        "Sparse regression with feasible dual bounds",
        "ISTA, FISTA and ADMM solve the same convex objective. A dual-feasible residual gives a real suboptimality bound. ADMM factorization and gradient iterations have different costs.",
        [
            (
                "Closed-form oracle",
                'from optimization_lab.first_order import lasso\nb=np.array([3.,-2.,.25])\nfor method in ["ista","fista","admm"]:\n    result=lasso(np.eye(3),b,1.,method)\n    assert np.allclose(result["x"],[2.,-1.,0.],atol=1e-6)\n    print(method,result["certificate"])',
            ),
            (
                "All seeds",
                'rows=[]\nfor seed in [0,1,2]:\n    for row in read(f"results/reference/first_order/lasso_seed{seed}/metrics.json")["rows"]:\n        rows.append(dict(seed=seed,method=row["method"],iterations=row["iterations"],**row["certificate"]))\npd.DataFrame(rows)',
            ),
            ("Iteration and time", 'figure("lasso_convergence_seed0")'),
        ],
    ),
    (
        "02_nuclear_completion",
        "Low-rank completion and a spectral dual certificate",
        "A masked squared-error objective with a uniform nuclear penalty is convex. Exact SVD shrinkage and feasible spectral dual bounds differ from the degree-regularized biased MovieLens objective.",
        [
            (
                "Missing entries are excluded",
                'from optimization_lab.first_order import nuclear_completion\ny=np.array([[4.,np.nan],[np.nan,2.]])\nresult=nuclear_completion(y,np.isfinite(y),1.)\nassert np.allclose(result["x"],np.diag([3.,1.]))\nprint(result["certificate"])',
            ),
            (
                "Every synthetic fit",
                'rows=[]\nfor seed in [0,1,2]:\n    rows += [dict(seed=seed,method=r["method"],heldout_rmse=r["unobserved_rmse_to_synthetic_truth"],gap=r["certificate"]["gap"]) for r in read(f"results/reference/first_order/nuclear_seed{seed}/metrics.json")["rows"]]\npd.DataFrame(rows)',
            ),
            ("Reconstruction evidence", 'figure("nuclear_completion_seed0")'),
        ],
    ),
    (
        "03_poisson",
        "Poisson rates at active boundaries",
        "Zero counts permit exact zero-rate optima. The supplied second day is validation, while the synthetic studies reserve an independent test day. Strict infinite likelihood and the declared predictive floor are reported separately.",
        [
            (
                "Boundary oracle",
                'from optimization_lab.first_order import poisson_prox,poisson_smooth\nassert poisson_prox(.5,0.,1.)==0.\nresult=poisson_smooth([0,4,0,2],0.)\nassert np.allclose(result["rate"],[0,4,0,2],atol=1e-5)\nprint(result["rate"],result["kkt_residual"])',
            ),
            (
                "Validation and test roles",
                'rows=[]\nfor label in ["original","synthetic_seed0","synthetic_seed1","synthetic_seed2"]:\n    m=read(f"results/reference/first_order/poisson_{label}/metrics.json")\n    rows.append(dict(task=label,**m["selected"]))\npd.DataFrame(rows)',
            ),
            ("Original counts and validation selection", 'figure("poisson_original")'),
            ("Independent synthetic task", 'figure("poisson_synthetic_seed0")'),
        ],
    ),
    (
        "04_conditioning",
        "Conditioning, acceleration and curvature access",
        "Known-optimum quadratics expose convergence under controlled condition numbers. Newton solves a Hessian system; CG accesses Hessian products. Their iteration counts do not mean equal information or cost.",
        [
            (
                "Newton analytical step",
                'a=np.diag([1.,1000.]); optimum=np.array([2.,-1.]); b=a@optimum\nx=np.linalg.solve(a,b)\nassert np.allclose(x,optimum)\nprint("One full-Hessian solve reaches the known optimum")',
            ),
            (
                "Retained convergence and failures",
                'm=read("results/reference/first_order/quadratic/metrics.json")\npd.DataFrame(m["rows"])',
            ),
            ("Measured histories", 'figure("quadratic_conditioning")'),
        ],
    ),
    (
        "05_speed",
        "Travel-time convexification and dual recovery",
        "Changing speed to segment time makes the fuel objective reciprocal-convex and the arrival windows linear. The full constant fuel term is retained. A nearly active nonnegative KKT fit refines the final feasible dual witness.",
        [
            (
                "Independent two-segment oracle",
                'from optimization_lab.barrier import speed_barrier\np=dict(d=[1.,2.],a=4.,b=3.,c=16.,smin=[.5,.5],smax=[5.,5.],tau_min=[0.,0.],tau_max=[10.,10.])\nr=speed_barrier(p)\nassert np.allclose(r["speed"],[2.,2.],atol=1e-5)\nassert abs(r["certificate"]["objective"]-57.)<1e-5\nprint(r["certificate"]["gap"])',
            ),
            (
                "Full original problem",
                'm=read("results/reference/convex/speed/metrics.json")\n{k:{q:v for q,v in m[k].items() if q!="multipliers"} for k in ["certificate","owned_certificate"]}',
            ),
            ("Physical schedule", 'figure("speed_schedule")'),
            ("Dual convergence", 'figure("speed_certificate")'),
        ],
    ),
    (
        "06_revenue_blending",
        "Revenue LP and geometric design blending",
        "Piecewise concave revenue has an LP hypograph and resource shadow prices. Circuit blending uses simplex weights over six supplied designs; performance bounds are conditional on log-convex physical metrics.",
        [
            (
                "Revenue witnesses",
                'm=read("results/reference/convex/revenue/metrics.json")\nassert m["duality_gap"]<1e-7\nm',
            ),
            (
                "Geometric rather than arithmetic interpolation",
                'values=np.array([4.,16.]); weights=np.array([.5,.5])\ngeometric=np.exp(np.log(values)@weights)\nassert abs(geometric-8.)<1e-12\nprint("Geometric blend",geometric,"arithmetic",values@weights)',
            ),
            (
                "Supplied-design programs",
                'pd.DataFrame(read("results/reference/convex/blend/metrics.json")["rows"])',
            ),
            (
                "Capacity and design plots",
                'figure("revenue_capacity")\nfigure("geometric_blending")',
            ),
        ],
    ),
    (
        "07_matrix_cones",
        "PSD projection and convex domains",
        "Symmetry is an explicit matrix-cone requirement. The antisymmetric input component contributes a constant loss. Perspective and square-root domains accompany transformed convex inequalities.",
        [
            (
                "Nonsymmetric input",
                'from optimization_lab.convex_cases import nearest_psd\na=np.array([[1.,3.],[-1.,-2.]])\nr=nearest_psd(a)\np=r["arrays"]["projection"]\nassert np.allclose(p,p.T) and np.linalg.eigvalsh(p).min()>-1e-10\nprint(p)',
            ),
            (
                "Domain witnesses",
                'm=read("results/reference/geometry/metrics.json")\nassert max(r["maximum_violation"] for r in m["rows"])<1e-7\npd.DataFrame(m["rows"])',
            ),
            ("Original PSD example", 'figure("psd_projection")'),
        ],
    ),
    (
        "08_regulator",
        "Four norms for finite-horizon control",
        "The supplied endpoint-constrained system has four distinct control norms. A terminal-controllability SVD independently verifies minimum energy; a HiGHS LP verifies elementwise L1. Nonunique solutions are compared by objective and feasibility.",
        [
            (
                "One-step analytical system",
                'from optimization_lab.convex_cases import regulator\np=dict(A=[[1.]],B=[[1.]],x_init=[1.],n=1,m=1,T=1)\nr=regulator(p)\nassert all(abs(row["clarabel"]["objective"]-1.)<1e-6 for row in r["rows"])\npd.DataFrame(r["rows"])',
            ),
            (
                "Original hundred-step study",
                'm=read("results/reference/convex/regulator/metrics.json")\nassert max(r["forward_simulation_terminal_residual"] for r in m["rows"])<1e-7\npd.DataFrame(m["rows"])',
            ),
            ("Controls and state trajectories", 'figure("regulator_norms")'),
        ],
    ),
    (
        "09_evacuation",
        "Conservation, capacity and terminal requirements",
        "The original signed-flow objective has no mandatory full terminal evacuation. A separate extension adds terminal safe-node mass. Both retain the same incidence balance and capacity conventions.",
        [
            (
                "Independently check population conservation",
                'a=np.load(ROOT/"assets/source_data/convex/evacuation.npz")\nfor name in ["original","terminal"]:\n    q=a[name+"_population"]\n    assert np.max(abs(q.sum(0)-1))<1e-8\n    assert np.max(q-a["node_capacity"][:,None])<1e-8\nprint("Mass and capacity checks passed for both programs")',
            ),
            (
                "Independent solver agreement",
                'pd.DataFrame(read("results/reference/convex/evacuation/metrics.json")["rows"])',
            ),
            ("Population evidence", 'figure("evacuation_population")'),
        ],
    ),
    (
        "10_scenario_risk",
        "Worst-scenario risk and a dual mixture",
        "The supplied theoretical model maximizes mean return minus worst-scenario quadratic risk. It permits shorting; gross exposure is reported rather than hidden. The common riskless asset permits an independent dual upper bound.",
        [
            (
                "Supplied covariance audit",
                'a=np.load(ROOT/"assets/source_data/convex/portfolio.npz")\nmatrices=a["scenario_covariances"]\nassert np.linalg.eigvalsh(matrices).min()>-1e-10\nassert np.allclose(matrices[:,0],0)\nprint("PSD scenarios with a common riskless asset:",matrices.shape)',
            ),
            (
                "Dual weights and exposure",
                'rows=read("results/reference/convex/portfolio/metrics.json")["rows"]\nfor row in rows:\n    assert abs(sum(row["scenario_dual_weights"])-row["gamma"])<1e-8\n    assert row["duality_gap"]<1e-6\npd.DataFrame(rows)[["gamma","risk_model","primal_objective","duality_gap","gross_exposure"]]',
            ),
            ("Declared theoretical tradeoffs", 'figure("scenario_risk")'),
        ],
    ),
    (
        "11_spectral",
        "Multiplicity, SVD and PCA geometry",
        "Householder tridiagonalization precedes shifted QR deflation. Schur columns are not general eigenvectors for nonnormal matrices. SVD verifies raw approximation identities while display values are clipped; PCA compares centered raw and standardized units.",
        [
            (
                "Repeated eigenvalues are retained",
                'from optimization_lab.spectral import symmetric_qr,nonsymmetric_schur\na=np.diag([0.,0.,2.,2.])\nr=symmetric_qr(a)\nassert np.allclose(r["values"],[0.,0.,2.,2.])\nprint(r["values"])\ns=nonsymmetric_schur(np.array([[1.,100.],[0.,2.]]))\nprint(s["scope"])',
            ),
            (
                "Raw SVD identity",
                'rows=read("results/reference/spectral/metrics.json")["compression"]\nfor row in rows:\n    assert np.isclose(row["raw_squared_error"],row["eckart_young_tail_sum"],rtol=1e-10)\npd.DataFrame(rows)',
            ),
            ("Licensed image approximations", 'figure("svd_gallery")\nfigure("svd_frontier")'),
            (
                "Centered PCA subspaces",
                'm=read("results/reference/spectral/metrics.json")\nassert max(r["sklearn_projector_difference"] for r in m["pca"])<1e-12\nfigure("pca_units")',
            ),
        ],
    ),
    (
        "12_recommendation",
        "Main project: regularized MovieLens factorization",
        "ALS, SGD and Adam share a degree-scaled objective at each penalty. All methods use the same fit / validation / official test split. The initial overfitting result is preserved, and the validation extension records every candidate. Final-test reuse and the limited three-seed scope are explicit.",
        [
            (
                "An ALS block is ridge regression",
                'neighbors=np.array([[1.,0.],[0.,1.],[1.,1.]])\ndesign=np.column_stack([np.ones(3),neighbors]);target=np.array([1.,2.,3.])\npenalty=np.array([.03,.1,.1])*3\ncoef=np.linalg.solve(design.T@design+np.diag(penalty),design.T@target)\nassert np.linalg.norm(design.T@(design@coef-target)+penalty*coef)<1e-12\nprint("Bias and factor block coefficients",coef)',
            ),
            (
                "Every selected model and seed",
                'fits=pd.read_csv(ROOT/"docs/recommendation_fits.csv")\nassert len(fits)==30\nfits[fits.protocol=="recommendation"].groupby("method").test_rmse.agg(["mean","std","min","max"])',
            ),
            (
                "Optimization versus generalization",
                'figure("recommendation_comparison")\nfigure("recommendation_matched_penalties")\nfigure("recommendation_optimization")',
            ),
            (
                "Fresh checkpoint evidence",
                'audit=read("docs/checkpoint_audit.json")\nassert audit["status"]=="passed" and audit["selected_models"]==30\npd.DataFrame(audit["rows"])[["family","method","seed","test_rmse","gradient_infinity_norm"]].head(10)',
            ),
        ],
    ),
    (
        "13_routing",
        "Main project: annealing, genetic search and ant colonies",
        "Both original TSPLIB instances compare three annealing neighborhoods, four pure GA recipes, AS / ACS / ACS+2opt, NN+2opt, two memetic GA extensions and OR-Tools. Every fit is retained. Work caps and library wall-clock budgets have different meanings.",
        [
            (
                "Permutation operators",
                'from optimization_lab.routing import order_crossover,cycle_crossover,mutate,tour_length\nrng=np.random.default_rng(0);a=rng.permutation(12);b=rng.permutation(12)\nfor child in [order_crossover(a,b,rng),cycle_crossover(a,b),mutate(a,rng,"insert")]:\n    assert np.array_equal(np.sort(child),np.arange(12))\nprint("Every child is a complete permutation")',
            ),
            (
                "All runs and gaps",
                'fits=pd.read_csv(ROOT/"docs/routing_fits.csv")\nassert len(fits)==84\nfits.groupby(["instance","method"]).gap_to_published_optimum_percent.agg(["mean","std","min","max"])',
            ),
            ("Measured comparison", 'figure("routing_comparison")'),
            ("Fixed-seed route gallery", 'figure("routing_fixed_seed_tours")'),
            ("Convergence within counted work", 'figure("routing_convergence")'),
        ],
    ),
    (
        "14_certificates",
        "Independent route certificates and release evidence",
        "A tour is a feasible upper bound; a minimum modified-cost 1-tree supplies an exact integer lower bound. Published TSPLIB lengths are evaluation references. An exact small-instance result does not prove a larger heuristic optimal.",
        [
            (
                "Recompute both large-instance lower bounds",
                'from optimization_lab.routing import integer_one_tree\nfor name in ["berlin52","ch150"]:\n    a=np.load(ROOT/f"assets/source_data/routing/{name}.npz")\n    certificate=integer_one_tree(a["distance"],a["one_tree_potentials"])\n    m=read(f"results/reference/routing/{name}/metrics.json")\n    assert certificate["bound"]==m["one_tree_certificate"]["bound"]<=m["published_optimum"]\n    print(name,certificate["bound"],m["published_optimum"])',
            ),
            (
                "Two independent exact methods",
                'from optimization_lab.routing import held_karp\nfrom optimization_lab.exact import mtz_tsp\na=np.load(ROOT/"assets/source_data/exact/tsp12.npz")\ndp=held_karp(a["distance"]);milp=mtz_tsp(a["distance"])\nassert dp["length"]==milp["length"]==2248 and milp["proven_optimal"]\nprint("Exact length",dp["length"],"MILP gap",milp["mip_gap"])',
            ),
            ("Integer bound histories", 'figure("routing_lower_bounds")'),
            (
                "Complete protocol counts",
                "from optimization_lab.reporting import records\nprotocols,counts=records()\nassert sum(counts.values())==31\nprint(counts)",
            ),
        ],
    ),
]


def main():
    root = Path(__file__).resolve().parents[1]
    for filename, title, introduction, sections in STUDIES:
        notebook = nbf.v4.new_notebook()
        notebook.metadata.update(
            kernelspec=dict(display_name="Python3", language="python", name="python3"),
            language_info=dict(name="python"),
        )
        notebook.cells = [
            nbf.v4.new_markdown_cell(
                f"# {title}\n\n{introduction}\n\nSee [theory](../docs/theory.md), [protocol](../docs/protocol.md) and [limitations](../docs/model_cards.md)."
            ),
            nbf.v4.new_code_cell(SETUP),
        ]
        for heading, code in sections:
            notebook.cells.extend(
                [nbf.v4.new_markdown_cell("## " + heading), nbf.v4.new_code_cell(code)]
            )
        notebook.cells.append(
            nbf.v4.new_markdown_cell(
                "Full training / search commands and cache provenance are in [reproduction](../docs/reproduction.md). This notebook inspects actually completed evidence and small analytical checks; opening it does not launch the full fitting campaign."
            )
        )
        nbf.write(notebook, root / "notebooks" / f"{filename}.ipynb")
    print("Created", len(STUDIES), "English walkthroughs")


if __name__ == "__main__":
    main()
