"""Create three English walkthroughs for certified imaging and modern model search."""

from pathlib import Path

import nbformat as nbf

SETUP = """from pathlib import Path
import json, sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import Image, display
ROOT=Path.cwd()
if ROOT.name == "notebooks": ROOT=ROOT.parent
sys.path.insert(0,str(ROOT/"src"))
def read(name): return json.loads((ROOT/name).read_text())
"""
STUDIES = [
    (
        "15_primal_dual",
        "Certified image reconstruction",
        "Periodic isotropic TV supplies a useful bridge between convex duality, first-order algorithms and a real image task. A stopping certificate does not establish that smoothing improved the unknown image.",
        [
            (
                "Verify the adjoint",
                'from optimization_lab.primal_dual import gradient, adjoint, denoise\nrng=np.random.default_rng(12)\nx=rng.normal(size=(8,7)); p=rng.normal(size=(2,8,7))\nassert abs(np.sum(gradient(x)*p)-np.sum(x*adjoint(p))) < 1e-12\nprint("Periodic adjoint identity verified")',
            ),
            (
                "Independent cone oracle",
                'from optimization_lab.primal_dual_study import conic_reference\nb=rng.normal(size=(8,7))\noracle, reference=conic_reference(b,.18)\nrows=[]\nfor method in ["pdhg","accelerated_pdhg","dual_fista","admm"]:\n    result=denoise(b,.18,method,steps=12000,gap_per_pixel=2e-9)\n    assert result["certificate"]["gap"] >= -1e-10\n    assert abs(result["certificate"]["primal"]-oracle["objective"]) < 1e-6\n    rows.append(dict(method=method,iterations=result["iterations"],gap=result["certificate"]["gap"]))\npd.DataFrame(rows)',
            ),
            (
                "All measured solves",
                'frame=pd.read_csv(ROOT/"docs/primal_dual_fits.csv")\nassert len(frame)==72\nframe.groupby(["dataset","penalty","method"])[["psnr","iterations","gap_per_pixel"]].mean()',
            ),
            (
                "Real-photo convergence",
                'display(Image(filename=str(ROOT/"assets/figures/primal_dual_photograph.png")))',
            ),
        ],
    ),
    (
        "16_adaptive_search",
        "Search algorithms and held-out decisions",
        "Sequential TPE and random search propose parameters for the same audited factor model. The common initial proposals, budget, selection role and unfavorable results make this a meaningful search comparison.",
        [
            (
                "Check shared startup proposals",
                'histories={}\nfor seed in [0,1,2]:\n    for strategy in ["random","tpe"]:\n        metrics=read(f"results/reference/adaptive_search/{strategy}/seed{seed}/metrics.json")\n        histories[strategy,seed]=metrics\n    assert [t["parameters"] for t in histories["random",seed]["trials"][:8]] == [t["parameters"] for t in histories["tpe",seed]["trials"][:8]]\nprint("All three shared startup sequences match")',
            ),
            (
                "Recompute validation selection",
                'for key,metrics in histories.items():\n    selected=min(metrics["trials"],key=lambda t:(t["validation_rmse"],t["number"]))\n    assert selected["number"] == metrics["selected_trial"]\n    assert selected["parameters"] == metrics["selected_parameters"]\n    assert len(metrics["trials"]) == 16\nprint("All six winners selected without test metrics")',
            ),
            (
                "Every selected deployment result",
                'frame=pd.read_csv(ROOT/"docs/adaptive_search_fits.csv")\nassert len(frame)==6\nframe.groupby("strategy")[["validation_rmse","test_rmse"]].agg(["mean","std"])',
            ),
            (
                "Validation trajectories and interpretation",
                'display(Image(filename=str(ROOT/"assets/figures/adaptive_search.png")))\nprint("A small density-model search need not beat random search. Epoch selection and parameter selection share validation; the official test remains reserved until each campaign completes.")',
            ),
        ],
    ),
    (
        "17_geometry_and_certificates",
        "Feasibility, certificates and information",
        "Mathematical certificates answer precise objective questions. Feasible domains, numerical residuals, held-out scores and computational budgets answer different questions and should be inspected together.",
        [
            (
                "A live domain point",
                'def perspective_feasible(x,y): return y>0 and x*x/y <= x-2*y*y+5\nassert perspective_feasible(1.,1.)\nassert not perspective_feasible(1.,0.)\nassert not perspective_feasible(3.,3.)\nwitnesses=read("results/reference/geometry/metrics.json")\npd.DataFrame([dict(domain=r["domain"],witness=r["witness"],violation=r["maximum_violation"]) for r in witnesses["rows"]])',
            ),
            (
                "Inspect a projected feasible dual",
                'from optimization_lab.primal_dual import certificate\nb=np.random.default_rng(3).normal(size=(5,6))\ncert=certificate(b,b,np.full((2,5,6),1e8),.2)\nassert cert["dual_feasibility"] <= 1e-12\nassert cert["gap"] >= -1e-12\ncert',
            ),
            (
                "Exact route interval",
                'from optimization_lab.routing import integer_one_tree,tour_length\ndata=np.load(ROOT/"assets/source_data/routing/berlin52.npz",allow_pickle=False)\nbound=integer_one_tree(data["distance"],data["one_tree_potentials"])["bound"]\nupper=tour_length(data["distance"],data["acs_2opt_seed0_tour"])\nassert bound <= 7542 <= upper\nprint("Exact integer bounds:",bound,"<= optimum <=",upper)',
            ),
            (
                "Compare evidence roles",
                'audit=read("docs/extensions_validation.json")\nassert audit["status"]=="passed"\nprint("72 feasible image solves and 96 real-data search candidates verified")\nprint("Open docs/explorer.html locally to inspect eight interactive concept groups. A heuristic route is an upper bound; validation selection is not a numerical optimality certificate.")',
            ),
        ],
    ),
]
for name, title, introduction, sections in STUDIES:
    cells = [nbf.v4.new_markdown_cell(f"# {title}\n\n{introduction}"), nbf.v4.new_code_cell(SETUP)]
    for heading, code in sections:
        cells.extend([nbf.v4.new_markdown_cell("## " + heading), nbf.v4.new_code_cell(code)])
    cells.append(
        nbf.v4.new_markdown_cell(
            "## Exercise and worked interpretation\n\nChange the modeling or budget assumption and identify which claim survives. A feasible TV dual remains a lower bound for its stated objective; it does not guarantee improved PSNR. A validation winner remains the winner of its declared candidates; a larger search can still overfit selection. A route certificate uses the exact integer metric, while a visual map is only a display. Keep these claims separate when interpreting the result."
        )
    )
    notebook = nbf.v4.new_notebook(
        cells=cells,
        metadata=dict(kernelspec=dict(display_name="Python 3", language="python", name="python3")),
    )
    nbf.write(notebook, Path("notebooks") / (name + ".ipynb"))
    print("Created", name)
